"use strict";
(() => {
  if (!window.SRT_REVIEW) return;
  const initial = window.SRT_REVIEW.feedback;
  const clone = value => JSON.parse(JSON.stringify(value));
  const canonical = value => JSON.stringify(value, Object.keys(value).sort());
  const namespace = [initial.base.corpus_id, initial.base.corpus_sha256,
    initial.base.review_id, initial.base.run_id, initial.base.evaluation_id].join(":");
  const status = document.getElementById("save-status");
  const reviewer = document.getElementById("reviewer");
  const controls = ["reviewer", "decision", "notes", "completed", "import-feedback", "resume-feedback", "export-feedback", "new-session"]
    .map(id => document.getElementById(id)).filter(Boolean);
  let draft = clone(initial), database = null, committed = 0, saving = false;
  let dirty = false, conflict = false, ready = false, selected = null, channel = null;
  const key = () => namespace + ":" + draft.session;
  const say = message => { status.textContent = message; };
  controls.forEach(control => { control.disabled = true; });
  const header = document.querySelector(".review-header");
  const menu = document.getElementById("feedback-menu");
  const measureHeader = () => document.documentElement.style.setProperty("--review-header-height", `${header.offsetHeight}px`);
  measureHeader();
  new ResizeObserver(measureHeader).observe(header);
  document.getElementById("resume-feedback").addEventListener("click", () => {
    menu.open = false;
    document.getElementById("import-feedback").click();
  });
  document.addEventListener("click", event => {
    if (!menu.contains(event.target)) menu.open = false;
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && menu.open) {
      menu.open = false;
      menu.querySelector("summary").focus();
    }
  });

  function validate(value) {
    if (!value || value.schema_version !== 1 || value.kind !== initial.kind ||
        canonical(value.base || {}) !== canonical(initial.base) ||
        typeof value.session !== "string" || !value.session ||
        !Number.isSafeInteger(value.edit_revision) || value.edit_revision < 0 ||
        !(value.reviewer === null || typeof value.reviewer === "string") ||
        !Array.isArray(value.entries) || value.entries.length !== initial.entries.length) {
      throw new Error("This feedback belongs to a different review or has an invalid format.");
    }
    const expected = new Map(initial.entries.map(entry => [entry.identifier, entry]));
    const seen = new Set();
    for (const entry of value.entries) {
      const base = expected.get(entry.identifier);
      if (!base || seen.has(entry.identifier) || entry.case_revision !== base.case_revision ||
          !["", "Correct", "Operator", "Caller", "Both speakers", "No speech", "Unsure"].includes(entry.decision) ||
          !["untouched", "draft", "completed"].includes(entry.state) || typeof entry.notes !== "string" ||
          (entry.state === "completed" && !entry.decision) ||
          (entry.state === "untouched" && (entry.decision || entry.notes))) {
        throw new Error("Feedback contains a stale, duplicate, or invalid block. Your current draft is unchanged.");
      }
      seen.add(entry.identifier);
    }
    return clone(value);
  }

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("srt-review-drafts", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("drafts");
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("Draft storage is blocked"));
      request.onsuccess = () => resolve(request.result);
    });
  }

  function transact(operation) {
    return new Promise((resolve, reject) => {
      const transaction = database.transaction("drafts", "readwrite");
      const store = transaction.objectStore("drafts");
      let result, failure;
      transaction.oncomplete = () => resolve(result);
      transaction.onabort = transaction.onerror = () => reject(failure || transaction.error || new Error("Draft save failed"));
      try { operation(store, value => { result = value; }, error => { failure = error; transaction.abort(); }); }
      catch (error) { failure = error; transaction.abort(); }
    });
  }

  function refresh() {
    reviewer.value = draft.reviewer || "";
    const completed = draft.entries.filter(entry => entry.state === "completed").length;
    const pending = draft.entries.filter(entry => entry.state === "draft").length;
    document.getElementById("review-progress").textContent =
      `${completed} / ${draft.entries.length} reviewed${pending ? ` · ${pending} drafts` : ""}`;
    if (!selected) return;
    const entry = draft.entries.find(item => item.identifier === selected);
    if (!entry) return;
    document.getElementById("edit-block-id").textContent = selected;
    document.getElementById("decision").value = entry.decision;
    document.getElementById("notes").value = entry.notes;
    document.getElementById("completed").checked = entry.state === "completed";
    document.getElementById("block-review-state").textContent =
      {untouched: "Not reviewed", draft: "Draft — not completed", completed: "Review completed"}[entry.state];
  }

  async function save() {
    if (saving || !dirty || conflict) return;
    if (!database) { say("Changes are only held in this page until exported. Browser storage is unavailable."); return; }
    saving = true;
    try {
      while (dirty && !conflict) {
        const snapshot = clone(draft), storageKey = key(), expected = committed;
        say("Saving…");
        await transact((store, done, fail) => {
          const request = store.get(storageKey);
          request.onsuccess = () => {
            const prior = request.result;
            if ((prior ? prior.edit_revision : 0) !== expected) {
              conflict = true;
              fail(new Error("Another tab changed this draft. Download your work or keep it as a separate draft."));
              return;
            }
            store.put(snapshot, storageKey);
            done(true);
          };
        });
        committed = snapshot.edit_revision;
        dirty = draft.edit_revision !== committed;
        if (channel) channel.postMessage({session: snapshot.session, revision: committed});
      }
      if (!conflict) say("Saved in this browser");
    } catch (error) {
      if (!conflict) { database.close(); database = null; }
      say(conflict ? error.message : "Save failed. Changes are only held in this page until exported. You can keep editing.");
    } finally { saving = false; }
  }

  function changed() {
    draft.edit_revision += 1;
    draft.updated_at = new Date().toISOString();
    dirty = true;
    refresh();
    void save();
  }

  document.addEventListener("srt-block-selected", event => {
    selected = event.detail;
    refresh();
  });
  reviewer.addEventListener("input", () => { draft.reviewer = reviewer.value || null; changed(); });
  for (const id of ["decision", "notes", "completed"]) {
    const control = document.getElementById(id);
    if (!control) continue;
    control.addEventListener(id === "notes" ? "input" : "change", () => {
      const entry = draft.entries.find(item => item.identifier === selected);
      if (!entry) return;
      entry.decision = document.getElementById("decision").value;
      entry.notes = document.getElementById("notes").value;
      const checked = document.getElementById("completed").checked;
      entry.state = checked && entry.decision ? "completed" : (entry.decision || entry.notes ? "draft" : "untouched");
      changed();
    });
  }

  document.getElementById("export-feedback").addEventListener("click", () => {
    if (!ready) return;
    const blob = new Blob([JSON.stringify(draft, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `review-feedback-${draft.session}.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  });

  document.getElementById("import-feedback").addEventListener("change", async event => {
    try {
      const file = event.target.files[0];
      if (!file) return;
      if (file.size > 10_000_000) throw new Error("Feedback file is too large.");
      const incoming = validate(JSON.parse(await file.text()));
      const local = new Map(draft.entries.map(entry => [entry.identifier, entry]));
      for (const entry of incoming.entries) {
        const existing = local.get(entry.identifier);
        if (existing.state !== "untouched" && entry.state !== "untouched" && canonical(existing) !== canonical(entry)) {
          throw new Error("Conflicting edits. Export this draft, then import into a fresh review session; neither version was changed.");
        }
      }
      if (draft.reviewer && incoming.reviewer && draft.reviewer !== incoming.reviewer) {
        throw new Error("Reviewer identities conflict. Current work was kept.");
      }
      if (saving) throw new Error("A save is in progress. Please import again after it finishes.");
      const merged = incoming.entries.map(entry => entry.state === "untouched" ? local.get(entry.identifier) : entry);
      draft.entries = clone(merged);
      draft.reviewer = draft.reviewer || incoming.reviewer;
      draft.imports = [...(draft.imports || []), {session: incoming.session, edit_revision: incoming.edit_revision}];
      changed();
    } catch (error) { say(error.message); }
    finally { event.target.value = ""; }
  });

  document.getElementById("new-session").addEventListener("click", () => {
    menu.open = false;
    if (saving) { say("Please wait for the current save before creating a separate draft."); return; }
    draft = clone(draft);
    draft.session = crypto.randomUUID();
    draft.edit_revision = 0; committed = 0; conflict = false;
    try { sessionStorage.setItem(namespace, draft.session); } catch (_) { /* Export remains available. */ }
    changed();
  });
  window.addEventListener("beforeunload", event => {
    if (dirty || conflict) { event.preventDefault(); event.returnValue = ""; }
  });

  async function initialize() {
    try {
      try { draft.session = sessionStorage.getItem(namespace) || draft.session; } catch (_) { /* Storage probe below is authoritative. */ }
      database = await openDatabase();
      const probe = "probe:" + crypto.randomUUID();
      await transact((store, done, fail) => {
        store.put("probe", probe);
        const request = store.get(probe);
        request.onsuccess = () => {
          if (request.result !== "probe") { fail(new Error("Storage probe failed")); return; }
          store.delete(probe); done(true);
        };
      });
      const saved = await transact((store, done) => { store.get(key()).onsuccess = event => done(event.target.result); });
      if (saved) { draft = validate(saved); committed = draft.edit_revision; }
      say(saved ? "Saved in this browser" : "No edits yet · autosave ready");
      if (typeof BroadcastChannel !== "undefined") {
        channel = new BroadcastChannel(namespace);
        channel.onmessage = event => {
          if (event.data.session === draft.session && event.data.revision !== committed) {
            conflict = true;
            say("Another tab changed this draft. Download your work or keep it as a separate draft; neither copy will be overwritten.");
          }
        };
      }
    } catch (_) {
      if (database) database.close();
      database = null;
      say("Browser storage is unavailable or incompatible. Changes are only held in this page until exported. Existing stored drafts were not changed.");
    } finally {
      ready = true;
      controls.forEach(control => { control.disabled = false; });
      refresh();
    }
  }
  void initialize();
})();
