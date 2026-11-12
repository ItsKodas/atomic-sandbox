//! The workspace (docking-UI redesign, `docs/superpowers/specs/2026-11-03-docking-ui-design.md`):
//! the 3D view and the panels as one tree of splits and tabbed leaves that the user rearranges,
//! replacing the two fixed docks. Display state only — nothing here is an `Action` (spec D-13).
//!
//! - `layout`: the tree and the small text grammar the profile saves it in (spec D-8, D-9);
//! - `tree`: the tree on screen (`egui_dock`), the 3D view's tile, the tabs, and [`Workspace`],
//!   which `UiState::workspace` holds (spec D-2..D-5);
//! - `autohide`: the unpinned panels' strips and flyout round the 3D view (spec D-10, D2);
//! - `presets`: the four built-in layouts of Window ▸ Layouts (spec D-11, D2);
//! - `detached`: panels in their own OS windows — the trees, moving panels between windows,
//!   docking back, the requests to the window side (`crate::detached`) and the profile's
//!   `window = …` lines (spec D-12, D3);
//! - `drag`: a tab dragged onto another window or out of every window (spec D-15; nothing floats
//!   inside a window, spec D-5).
pub mod autohide;
pub mod detached;
pub mod drag;
pub mod layout;
pub mod presets;
pub mod tree;

pub use tree::Workspace;
