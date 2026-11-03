//! The workspace (docking-UI redesign, `docs/superpowers/specs/2026-11-03-docking-ui-design.md`):
//! the 3D view and the panels as one tree of splits and tabbed leaves that the user rearranges,
//! replacing the two fixed docks. Display state only — nothing here is an `Action` (spec D-13).
//!
//! - `layout`: the tree and the small text grammar the profile saves it in (spec D-8, D-9);
//! - `tree`: the tree on screen (`egui_dock`), the 3D view's tile, the tabs, and [`Workspace`],
//!   which `UiState::workspace` holds (spec D-2..D-5);
//! - `autohide`: the unpinned panels' strips and flyout round the 3D view (spec D-10, D2).
pub mod autohide;
pub mod layout;
pub mod tree;

pub use tree::Workspace;
