//! The Git (Source Control) dock panel: async git/gh runners and the panel UI.

use std::path::Path;
use std::sync::mpsc::Receiver;

use crate::theme::ACCENT_GOLD;

/// State for the Git (Source Control) dock panel.
#[derive(Default)]
pub(crate) struct GitUi {
    pub(crate) commit_msg: String,
    /// Output of the most recent operation (with a trailing status summary).
    pub(crate) output: String,
    /// Receiver for an in-flight async git operation (None = idle).
    pub(crate) rx: Option<Receiver<String>>,
    /// Wizard: desired GitHub repo name (seeded from the project folder once).
    pub(crate) gh_repo_name: String,
    /// Wizard: create the GitHub repo private (vs public).
    pub(crate) gh_private: bool,
    /// Whether `gh_repo_name` has been seeded from the folder name yet.
    pub(crate) seeded: bool,
}

/// The current branch from `.git/HEAD` (a cheap file read — no subprocess, safe
/// to call each frame). Returns the branch name, a short commit hash in `()` when
/// detached, or None if `root` isn't a git repo / HEAD is unreadable.
pub(crate) fn current_branch(root: &Path) -> Option<String> {
    let head = std::fs::read_to_string(root.join(".git").join("HEAD")).ok()?;
    let head = head.trim();
    if let Some(branch) = head.strip_prefix("ref: refs/heads/") {
        Some(branch.to_string())
    } else if head.len() >= 7 {
        Some(format!("({})", &head[..7]))
    } else {
        None
    }
}

/// Run a program in `project`, capturing stdout+stderr as one string.
pub(crate) fn run_tool(program: &str, project: &Path, args: &[&str]) -> String {
    match std::process::Command::new(program)
        .current_dir(project)
        .args(args)
        .output()
    {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            let e = String::from_utf8_lossy(&o.stderr);
            if !e.trim().is_empty() {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&e);
            }
            if s.trim().is_empty() {
                s = format!("(ok: {program} {})", args.join(" "));
            }
            s
        }
        Err(e) => format!("Couldn't run {program}: {e}\n(Is {program} installed and on PATH?)"),
    }
}

/// Run a git command in `project`.
pub(crate) fn git_run(project: &Path, args: &[&str]) -> String {
    run_tool("git", project, args)
}

/// Run `f` (one or more git commands) on a background thread, then append a short
/// `git status -sb` summary; the result lands in `git.output` (polled in update).
pub(crate) fn git_async(git: &mut GitUi, project: &Path, f: impl FnOnce(&Path) -> String + Send + 'static) {
    let project = project.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    git.rx = Some(rx);
    std::thread::spawn(move || {
        let mut out = f(&project);
        out.push_str("\n\n— status —\n");
        out.push_str(&git_run(&project, &["status", "-sb"]));
        let _ = tx.send(out);
    });
}

/// Full async flow to publish the project to a new GitHub repo: init (if needed),
/// commit, then `gh repo create … --push`.
pub(crate) fn github_publish(git: &mut GitUi, project: &Path) {
    let name = git.gh_repo_name.trim().to_string();
    let vis = if git.gh_private { "--private" } else { "--public" };
    git_async(git, project, move |p| {
        let mut out = String::new();
        if !p.join(".git").exists() {
            out.push_str("$ git init\n");
            out.push_str(&git_run(p, &["init"]));
            out.push_str("\n\n");
        }
        out.push_str("$ git add -A\n");
        out.push_str(&git_run(p, &["add", "-A"]));
        out.push_str("\n\n$ git commit -m \"Initial commit\"\n");
        out.push_str(&git_run(p, &["commit", "-m", "Initial commit"]));
        out.push_str(&format!(
            "\n\n$ gh repo create {name} --source=. {vis} --push\n"
        ));
        out.push_str(&run_tool(
            "gh",
            p,
            &["repo", "create", &name, "--source=.", vis, "--push"],
        ));
        out
    });
}

/// Render the Git panel: a setup wizard when the project isn't a repo (initialize
/// locally, or create it on GitHub), and the status / commit / push panel once it
/// is. All git/gh calls run asynchronously so the UI never freezes.
pub(crate) fn git_tab_ui(ui: &mut egui::Ui, git: &mut GitUi, project: Option<&Path>) {
    let Some(project) = project else {
        ui.centered_and_justified(|ui| {
            ui.label(egui::RichText::new("Open or create a project to use Git.").weak());
        });
        return;
    };

    // Seed the GitHub repo-name field from the project folder name (once).
    if !git.seeded {
        git.gh_repo_name = project
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        git.seeded = true;
    }

    let is_repo = project.join(".git").exists();
    // Auto-refresh status the first time the panel is shown for a repo.
    if is_repo && git.output.is_empty() && git.rx.is_none() {
        git_async(git, project, |p| git_run(p, &["status"]));
    }
    let busy = git.rx.is_some();

    egui::TopBottomPanel::top("git_header").show_inside(ui, |ui| {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.heading("Git");
            if busy {
                ui.spinner();
                ui.label(egui::RichText::new("working…").weak());
            }
        });
        ui.add_space(4.0);

        if is_repo {
            // ---- Repository controls ----
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(!busy, egui::Button::new("Refresh")).clicked() {
                    git_async(git, project, |p| git_run(p, &["status"]));
                }
                if ui.add_enabled(!busy, egui::Button::new("Pull")).clicked() {
                    git_async(git, project, |p| git_run(p, &["pull"]));
                }
                if ui.add_enabled(!busy, egui::Button::new("Push")).clicked() {
                    git_async(git, project, |p| git_run(p, &["push"]));
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Message:");
                ui.add(
                    egui::TextEdit::singleline(&mut git.commit_msg)
                        .desired_width(f32::INFINITY)
                        .hint_text("Commit message"),
                );
            });
            let can_commit = !busy && !git.commit_msg.trim().is_empty();
            if ui
                .add_enabled(can_commit, egui::Button::new("Stage all & Commit"))
                .clicked()
            {
                let msg = git.commit_msg.clone();
                git_async(git, project, move |p| {
                    let a = git_run(p, &["add", "-A"]);
                    let b = git_run(p, &["commit", "-m", &msg]);
                    format!("$ git add -A\n{a}\n\n$ git commit -m \"{msg}\"\n{b}")
                });
                git.commit_msg.clear();
            }
        } else {
            // ---- Setup wizard (not a repo yet) ----
            ui.label(
                egui::RichText::new("This project isn't version-controlled yet.")
                    .color(ACCENT_GOLD),
            );
            ui.add_space(6.0);
            if ui
                .add_enabled(!busy, egui::Button::new("Initialize local repository"))
                .on_hover_text("git init + an initial commit")
                .clicked()
            {
                git_async(git, project, |p| {
                    let a = git_run(p, &["init"]);
                    let b = git_run(p, &["add", "-A"]);
                    let c = git_run(p, &["commit", "-m", "Initial commit"]);
                    format!("$ git init\n{a}\n\n$ git add -A\n{b}\n\n$ git commit\n{c}")
                });
            }

            ui.add_space(8.0);
            ui.separator();
            ui.label(egui::RichText::new("…or create it on GitHub").strong());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Repo name:");
                ui.add(
                    egui::TextEdit::singleline(&mut git.gh_repo_name)
                        .desired_width(200.0)
                        .hint_text("my-game"),
                );
                ui.selectable_value(&mut git.gh_private, false, "Public");
                ui.selectable_value(&mut git.gh_private, true, "Private");
            });
            let can_gh = !busy && !git.gh_repo_name.trim().is_empty();
            if ui
                .add_enabled(can_gh, egui::Button::new("Create on GitHub & push"))
                .clicked()
            {
                github_publish(git, project);
            }
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(
                    "Needs the GitHub CLI (gh) signed in. If it isn't, open the Terminal tab \
                     and run:  gh auth login",
                )
                .small()
                .italics()
                .weak(),
            );
        }
        ui.add_space(6.0);
    });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&git.output).monospace())
                        .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
    });
}
