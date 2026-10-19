"""Reference answers from PySCF for the app's Compare panel (optional; nothing links it).

The app (`crates/app/src/reference/`) embeds this file, writes it to a temporary directory and runs
it as a separate process with the current geometry on stdin. CLAUDE.md rule 2: reference codes are
test oracles only, never linked — this process never feeds the dynamics, it answers one geometry and
exits.

    python pyscf_ref.py --probe          -> {"ok": true, "pyscf": "2.x", ...}
    python pyscf_ref.py < request.json   -> the answer (below)

Request (JSON on stdin; atomic units: Bohr):
    {"z": [8, 1, 1], "positions": [[x, y, z], ...], "charge": 0, "multiplicity": null | 1 | 2 ...,
     "method": "lda" | "pbe" | "b3lyp" | "ccsd_t", "basis": "def2-TZVP", "optimise": false}

Answer: one line on stdout, `@@PYSCF_REF@@` followed by JSON (everything else the libraries print
goes to stderr):
    {"ok": true, "method": ..., "basis": ..., "energy": E (Ha), "forces": [[Fx, Fy, Fz], ...] (Ha/Bohr,
     −dE/dR) | null, "dipole": [mx, my, mz] (e·Bohr, about the coordinates' origin) | null,
     "optimised": [[x, y, z], ...] (Bohr) | null, "optimise_converged": bool | null,
     "notes": [...], "pyscf": version, "wall_s": seconds}
    {"ok": false, "error": "..."} on failure (exit code 1).

Methods: lda = Slater exchange + Perdew–Wang 92 correlation (LDA_X,LDA_C_PW — this program's own
LSDA), pbe, b3lyp (Kohn-Sham DFT; restricted for a singlet, unrestricted otherwise), ccsd_t =
CCSD(T) on a Hartree-Fock reference (energy; forces from the CCSD(T) analytic gradient where PySCF
has one — closed shell —, otherwise none; dipole from the unrelaxed CCSD density). "optimise" needs
the geomeTRIC (`pip install geometric`) or PyBerny (`pip install pyberny`) package.

Windows: PySCF publishes no Windows build; run it under WSL (the app tries `wsl -e python3`) or point
ATOMIC_SANDBOX_PYTHON at an interpreter that has it.
"""

import json
import sys
import time

MARKER = "@@PYSCF_REF@@"
XC = {"lda": "LDA_X,LDA_C_PW", "pbe": "PBE", "b3lyp": "B3LYP"}


def geomopt():
    """The optimiser module PySCF can drive, or None."""
    try:
        import geometric  # noqa: F401
        from pyscf.geomopt import geometric_solver

        return "geometric", geometric_solver
    except Exception:
        pass
    try:
        import berny  # noqa: F401
        from pyscf.geomopt import berny_solver

        return "pyberny", berny_solver
    except Exception:
        return None, None


def probe():
    import pyscf

    name, _ = geomopt()
    return {"ok": True, "pyscf": pyscf.__version__, "python": sys.version.split()[0], "geomopt": name}


def optimise(target, notes):
    """Relax with whichever optimiser is installed; returns (positions in Bohr, converged)."""
    name, solver = geomopt()
    if solver is None:
        raise RuntimeError("relaxing needs the geomeTRIC or PyBerny package: pip install geometric")
    notes.append(f"relaxed with {name}")
    try:
        conv, mol_eq = solver.kernel(target, maxsteps=100)
    except TypeError:
        mol_eq, conv = solver.optimize(target, maxsteps=100), None
    return mol_eq.atom_coords(unit="Bohr").tolist(), (None if conv is None else bool(conv))


def run(req):
    from pyscf import gto, scf, dft
    from pyscf.data.elements import ELEMENTS

    z = [int(x) for x in req["z"]]
    pos = [[float(c) for c in p] for p in req["positions"]]
    if not z or len(z) != len(pos):
        raise ValueError("the request needs one position per nucleus")
    charge = int(req.get("charge", 0))
    method = req.get("method", "lda")
    basis = req.get("basis") or "def2-TZVP"
    electrons = sum(z) - charge
    mult = req.get("multiplicity")
    spin = (int(mult) - 1) if mult else electrons % 2
    notes = []
    mol = gto.M(atom=[[ELEMENTS[a], tuple(p)] for a, p in zip(z, pos)], unit="Bohr", basis=basis, charge=charge, spin=spin, verbose=0)
    out = {"ok": True, "method": method, "basis": basis, "spin": spin, "forces": None, "dipole": None, "optimised": None, "optimise_converged": None}
    if method in XC:
        mf = (dft.RKS(mol) if spin == 0 else dft.UKS(mol))
        mf.xc = XC[method]
        mf.grids.level = 4
        mf.conv_tol = 1e-9
        mf.verbose = 0
        out["energy"] = float(mf.kernel())
        if not mf.converged:
            notes.append("SCF did not converge")
        notes.append(f"{'restricted' if spin == 0 else 'unrestricted'} Kohn-Sham, xc {XC[method]}, grid level 4")
        out["forces"] = (-mf.nuc_grad_method().kernel()).tolist()
        out["dipole"] = [float(x) for x in mf.dip_moment(unit="AU", verbose=0)]
        target = mf
    elif method == "ccsd_t":
        from pyscf import cc

        mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            notes.append("Hartree-Fock did not converge")
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        et = mycc.ccsd_t()
        out["energy"] = float(mycc.e_tot + et)
        notes.append(f"CCSD(T) on {'RHF' if spin == 0 else 'UHF'}, all electrons correlated")
        target = None
        try:
            from pyscf.grad import ccsd_t as ccsd_t_grad

            if spin != 0:
                raise NotImplementedError("open shell")
            g = ccsd_t_grad.Gradients(mycc)
            out["forces"] = (-g.kernel()).tolist()
            target = g.as_scanner()
        except Exception as e:  # no analytic CCSD(T) gradient for this case
            notes.append(f"no CCSD(T) forces ({type(e).__name__}: {e})")
        try:
            rdm1 = mycc.make_rdm1()
            if spin == 0:
                c = mf.mo_coeff
                dm = c @ rdm1 @ c.T
            else:
                dm = [mf.mo_coeff[s] @ rdm1[s] @ mf.mo_coeff[s].T for s in (0, 1)]
            out["dipole"] = [float(x) for x in mf.dip_moment(mol, dm, unit="AU", verbose=0)]
            notes.append("dipole from the unrelaxed CCSD density ((T) not included)")
        except Exception as e:
            notes.append(f"no dipole ({type(e).__name__}: {e})")
    else:
        raise ValueError(f"unknown method {method!r} (lda, pbe, b3lyp, ccsd_t)")
    if req.get("optimise"):
        if target is None:
            raise RuntimeError("relaxing needs forces, which this method did not give here")
        out["optimised"], out["optimise_converged"] = optimise(target, notes)
    out["notes"] = notes
    return out


def main():
    real = sys.stdout
    sys.stdout = sys.stderr  # the libraries' chatter must not reach the answer
    t0 = time.time()
    try:
        if "--probe" in sys.argv:
            out = probe()
        else:
            out = run(json.load(sys.stdin))
            import pyscf

            out["pyscf"] = pyscf.__version__
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out["wall_s"] = time.time() - t0
    real.write(MARKER + json.dumps(out) + "\n")
    real.flush()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
