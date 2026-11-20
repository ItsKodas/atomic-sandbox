# Models

Trained models (`.model` files) for the **estimated** solver, and the shared collections it learns into (`collections/`, this machine's own training data, never committed or packaged).

**No trained models ship any more.** The four that did (`hydrogen.model`, `h2o.model`, `h3+.model`, `SiH4.model`) were taught before the Fermi–Dirac switch (integer / shared occupations), and this version refuses such a file on load with the reason. They were removed with the switch rather than re-made (the user's decision of 2026-11-16): teach a model in the app instead. The estimated solver learns into the shared collection for the scene's settings by itself (a sampling pass, then every DFT answer it keeps), so nothing needs to be loaded first.

- Open a model file with **Load model…** in the Model tab (the dialog starts in this folder); **Save model…** and **Save a specialised model…** start here too.
- A model file only runs on the settings it was trained at: the grid (spacing and size), σ, charge, multiplicity, ionic potential, teacher and teacher tolerance. If the scene differs, a card lists the differences and offers to apply the model's settings.
- Name a file after its scene and grid, e.g. `h2-accurate.model`, `h2o-balanced.model`.
- GitHub refuses files over 100 MB.
- This README is the folder's only committed file. Keep it: a portable copy (`build.bat`, the release zip) finds its `models\` folder, and with it `models\collections`, by this folder existing beside the exe.
