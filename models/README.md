# Models

Trained models (`.model` files) for the **estimated** solver. They ship with the repository and in the release zip, so they are ready to load without training.

- Open one with **Load model…** in the Model tab (the dialog starts in this folder); **Save model…** starts here too.
- A model only runs on the settings it was trained at: the grid (spacing and size), σ, charge, multiplicity, ionic potential, teacher and teacher tolerance. If the scene differs, a card lists the differences and offers to apply the model's settings.
- Name a file after its scene and grid, e.g. `h2-accurate.model`, `h2o-balanced.model`.
- GitHub refuses files over 100 MB.
