// erf(x)/x and erfc for the ionic kernels (M7 Task 8a): a VERBATIM copy of nuclear.wgsl's
// TWO_OVER_SQRT_PI, SQRT_PI, the CH* Chebyshev coefficients, erfc_scaled, erfc_ge1, the S* series
// coefficients and erf_over_x (their accuracy analysis is in nuclear.wgsl's header). nuclear.wgsl itself is
// not touched, so the IonicPolicy::None path stays bitwise; this file is prepended to ionic.wgsl
// (kernels.rs SHADERS). Keep the two copies identical.

const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;
// f(x) = √π·x·e^{x²}·erfc(x) for x ≥ 1 (0.758 at x = 1, → 1 as x → ∞): on [1, 4) a degree-12
// Chebyshev series in t = (2x − 5)/3 (Clenshaw), from 4 on the asymptotic series in 1/x². Fit
// against f64 erfc (Task 10): |Δerf| ≤ 9.7e-10 with f64 coefficients, 6.1e-9 with these f32
// ones (the rounding of CH0 = 1.80 dominates; degree 14/16 change nothing), vs A&S 7.1.26's
// 1.4e-7; the asymptotic branch is 7e-6 relative at x = 4, i.e. 1e-13 absolute in erf.
const SQRT_PI: f32 = 1.7724538509055159;
const CH0: f32 = 1.8048557623e+00;
const CH1: f32 = 9.5036126435e-02;
const CH2: f32 = -3.4561538972e-02;
const CH3: f32 = 1.0820666035e-02;
const CH4: f32 = -3.0682293564e-03;
const CH5: f32 = 8.0784264219e-04;
const CH6: f32 = -2.0040632772e-04;
const CH7: f32 = 4.7291524862e-05;
const CH8: f32 = -1.0687072521e-05;
const CH9: f32 = 2.3243735801e-06;
const CH10: f32 = -4.8843173186e-07;
const CH11: f32 = 9.9470529692e-08;
const CH12: f32 = -1.9682356004e-08;

fn erfc_scaled(x: f32) -> f32 {
    if (x < 4.0) {
        let t = (2.0 * x - 5.0) / 3.0;
        let t2 = 2.0 * t;
        var b1 = 0.0;
        var b2 = 0.0;
        var b0 = 0.0;
        b0 = t2 * b1 - b2 + CH12; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH11; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH10; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH9; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH8; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH7; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH6; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH5; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH4; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH3; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH2; b2 = b1; b1 = b0;
        b0 = t2 * b1 - b2 + CH1; b2 = b1; b1 = b0;
        return t * b1 - b2 + 0.5 * CH0;
    }
    let u = 1.0 / (x * x);
    return 1.0 + u * (-0.5 + u * (0.75 + u * (-1.875 + u * (6.5625 + u * (-29.53125)))));
}

// erfc(x) = e^{−x²}·f(x)/(√π x) for x ≥ 1.
fn erfc_ge1(x: f32) -> f32 { return exp(-x * x) * erfc_scaled(x) / (SQRT_PI * x); }
// erf(x)/x = (2/√π) Σ_k c_k x^{2k},  c_k = (−1)^k / (k!(2k+1))
const S1: f32 = -0.3333333333333333;
const S2: f32 = 0.1;
const S3: f32 = -0.023809523809523808;
const S4: f32 = 0.004629629629629629;
const S5: f32 = -0.0007575757575757576;
const S6: f32 = 0.00010683760683760684;
const S7: f32 = -1.3227513227513228e-05;
const S8: f32 = 1.4589169000933706e-06;
const S9: f32 = -1.4503852223150468e-07;
const S10: f32 = 1.3122532963802806e-08;

// erf(x)/x for x ≥ 0.
fn erf_over_x(x: f32) -> f32 {
    if (x < 1.0) {
        let y = x * x;
        var s = S10;
        s = S9 + y * s;
        s = S8 + y * s;
        s = S7 + y * s;
        s = S6 + y * s;
        s = S5 + y * s;
        s = S4 + y * s;
        s = S3 + y * s;
        s = S2 + y * s;
        s = S1 + y * s;
        s = 1.0 + y * s;
        return TWO_OVER_SQRT_PI * s;
    }
    return (1.0 - erfc_ge1(x)) / x;
}
