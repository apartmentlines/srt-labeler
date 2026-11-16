async page => {
  const root = page.url();
  const browser = page.context().browser();
  const contexts = [];
  const evidence = [];
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const context = async (blocked = false) => {
    const created = await browser.newContext({acceptDownloads: true});
    contexts.push(created);
    if (blocked) await created.addInitScript(() => {
      Object.defineProperty(window, "indexedDB", {get() { throw new Error("fixture: storage blocked"); }});
    });
    return created;
  };
  const ready = async target => {
    await target.waitForFunction(() => !document.getElementById("reviewer").disabled);
  };
  const saved = async target => {
    await target.waitForFunction(() => document.getElementById("save-status").textContent.startsWith("Saved in this browser"));
  };
  const download = async (target, action) => {
    const [artifact] = await Promise.all([target.waitForEvent("download"), action()]);
    let content = "";
    for await (const chunk of await artifact.createReadStream()) content += chunk.toString();
    return JSON.parse(content);
  };
  const upload = async (target, value) => target.evaluate(record => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([JSON.stringify(record)], "feedback.json", {type: "application/json"}));
    const input = document.getElementById("import-feedback");
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", {bubbles: true}));
  }, value);
  try {
    const firstContext = await context();
    const first = await firstContext.newPage();
    await first.goto(root);
    await first.locator(".call-link").first().click();
    await ready(first);
    const callURL = first.url();
    const identifier = await first.locator("#edit-block-id").textContent();
    await first.locator("#reviewer").fill("Offline acceptance fixture");
    await first.locator("#decision").selectOption("Caller");
    await first.locator("#completed").check();
    const exported = await download(first, () => first.evaluate(() => {
      const notes = document.getElementById("notes");
      notes.value = "Latest unsaved edit <img src=x onerror=alert(1)>";
      notes.dispatchEvent(new Event("input", {bubbles: true}));
      document.getElementById("export-feedback").click();
    }));
    const entry = exported.entries.find(item => item.identifier === identifier);
    assert(entry.notes.startsWith("Latest unsaved edit") && entry.state === "completed", "Export lost pending editor state");
    await saved(first);
    await first.reload();
    await ready(first);
    assert(await first.locator("#notes").inputValue() === entry.notes, "Reload lost saved state");
    assert(await first.locator(".feedback-editor img").count() === 0, "Feedback became executable markup");
    evidence.push("Latest edits export before autosave; reload retains decisions and literal notes");
    await first.goto(root);
    await ready(first);
    assert(await first.locator("#reviewer").inputValue() === "Offline acceptance fixture", "Reviewer name was lost between pages");
    assert(await first.locator('a[href$=".xlsx"]').count() === 0, "Spreadsheet download remains in the review UI");
    await first.goto(callURL);
    await ready(first);
    assert(await first.locator(".review-header #reviewer").count() === 1, "Reviewer controls are outside the shared header");
    await first.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    assert(await first.locator(".review-header").evaluate(header => Math.abs(header.getBoundingClientRect().top) < 1), "Review header scrolled out of view");
    await first.locator("#feedback-menu summary").click();
    const chooser = first.waitForEvent("filechooser");
    await first.locator("#resume-feedback").click();
    await chooser;
    evidence.push("Shared header stays visible; reviewer survives navigation and Files opens feedback import");

    const second = await firstContext.newPage();
    await second.goto(callURL);
    await ready(second);
    await second.locator("#notes").fill("Competing tab version");
    await saved(second);
    await first.waitForFunction(() => document.getElementById("save-status").textContent.includes("Another tab"));
    assert(await first.locator("#notes").inputValue() === entry.notes, "Another tab overwrote local edits");
    await first.locator("#feedback-menu summary").click();
    await first.locator("#new-session").click();
    await saved(first);
    assert(await second.locator("#notes").inputValue() === "Competing tab version", "Fork overwrote the competing draft");
    evidence.push("Competing tabs detected; separate draft preserves both versions");

    const recoveryContext = await context();
    const recovery = await recoveryContext.newPage();
    await recovery.goto(callURL);
    await ready(recovery);
    await upload(recovery, exported);
    await saved(recovery);
    assert(await recovery.locator("#notes").inputValue() === entry.notes, "Import into empty storage lost feedback");
    const stale = JSON.parse(JSON.stringify(exported));
    stale.base.corpus_revision += 1;
    await upload(recovery, stale);
    await recovery.waitForFunction(() => document.getElementById("save-status").textContent.includes("different review"));
    const conflicting = JSON.parse(JSON.stringify(exported));
    conflicting.entries.find(item => item.identifier === identifier).notes = "conflicting import";
    await upload(recovery, conflicting);
    await recovery.waitForFunction(() => document.getElementById("save-status").textContent.includes("Conflicting edits"));
    assert(await recovery.locator("#notes").inputValue() === entry.notes, "Rejected import modified the draft");
    const duplicate = JSON.parse(JSON.stringify(exported));
    duplicate.entries[1] = duplicate.entries[0];
    await upload(recovery, duplicate);
    await recovery.waitForFunction(() => document.getElementById("save-status").textContent.includes("duplicate"));
    assert(await recovery.locator("#notes").inputValue() === entry.notes, "Duplicate import modified the draft");
    evidence.push("Portable recovery works; stale, conflicting and duplicate imports preserve existing work");

    const blockedContext = await context(true);
    const blocked = await blockedContext.newPage();
    await blocked.goto(callURL);
    await ready(blocked);
    await blocked.locator("#decision").selectOption("Unsure");
    await blocked.locator("#notes").fill("Memory-only recovery fixture");
    const memoryExport = await download(blocked, () => blocked.locator("#export-feedback").click());
    assert(memoryExport.entries.find(item => item.identifier === identifier).notes === "Memory-only recovery fixture", "Unavailable storage disabled export");
    assert((await blocked.locator("#save-status").textContent()).includes("only held in this page"), "Unavailable storage showed false saved status");
    evidence.push("Unavailable storage keeps editing and export available with truthful status");

    await recovery.evaluate(() => {
      IDBObjectStore.prototype.put = function () { throw new DOMException("fixture: quota", "QuotaExceededError"); };
    });
    await recovery.locator("#notes").fill("Write failed but this edit must survive");
    await recovery.waitForFunction(() => document.getElementById("save-status").textContent.includes("Save failed"));
    const failedExport = await download(recovery, () => recovery.locator("#export-feedback").click());
    assert(failedExport.entries.find(item => item.identifier === identifier).notes.startsWith("Write failed"), "Failed write lost export state");
    evidence.push("Later quota failure retains current draft in memory and export");

    await first.locator("#play-context").click();
    await first.waitForFunction(() => document.querySelector("audio").currentTime >= 0 && document.querySelector("audio").readyState >= 2);
    await first.locator("#pause-audio").click();
    await first.setViewportSize({width: 390, height: 844});
    assert(await first.locator(".mobile-controls").isVisible(), "Mobile playback controls are hidden");
    assert(await first.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), "Mobile layout overflows horizontally");
    evidence.push("Audio context playback and mobile controls remain usable");
    return {passed: evidence.length, evidence};
  } finally {
    for (const created of contexts) await created.close();
  }
}
