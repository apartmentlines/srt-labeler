"use strict";
(() => {
  for (const link of document.querySelectorAll("a[download]")) {
    const message = document.createElement("span");
    message.className = "download-status count";
    message.setAttribute("role", "status");
    link.after(message);
    let pending = false;
    link.addEventListener("click", async event => {
      event.preventDefault();
      if (pending) return;
      pending = true;
      link.setAttribute("aria-busy", "true");
      message.textContent = " Preparing download…";
      let url = null;
      try {
        const response = await fetch(link.href, {cache: "no-store", signal: AbortSignal.timeout(60000)});
        if (!response.ok) throw new Error("Download failed.");
        const contents = await response.blob();
        url = URL.createObjectURL(new Blob([contents], {type: "application/octet-stream"}));
        const save = document.createElement("a");
        save.href = url;
        save.download = link.download;
        document.body.append(save);
        save.click();
        save.remove();
        message.textContent = " Download started.";
      } catch (error) {
        message.textContent = " Download failed. Please try again.";
      } finally {
        if (url) setTimeout(() => URL.revokeObjectURL(url), 60000);
        pending = false;
        link.removeAttribute("aria-busy");
      }
    });
  }
  const audio = document.querySelector("audio");
  if (!audio) return;
  document.documentElement.classList.add("js-ready");
  const blocks = [...document.querySelectorAll(".block")];
  const reviewBlocks = blocks.filter(block => block.classList.contains("review"));
  const byId = new Map(blocks.map(block => [block.id, block]));
  const status = document.querySelector("#playback-status");
  let selected = null;
  let stopAt = null;
  let request = 0;
  let localAudio = null;
  let audioUrl = null;
  const timestamp = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

  function select(block) {
    if (!block) return;
    if (selected && selected !== block) {
      ++request;
      audio.pause();
      stopAt = null;
      status.textContent = "Block selected. Press Play to listen.";
    }
    if (selected) selected.classList.remove("selected");
    selected = block;
    selected.classList.add("selected");
    document.querySelector("#selected-id").textContent = block.dataset.noteId;
    document.querySelector("#mobile-id").textContent = block.dataset.noteId;
    document.querySelector("#selected-time").textContent = `${timestamp(Number(block.dataset.start))} – ${timestamp(Number(block.dataset.end))}`;
    const position = reviewBlocks.indexOf(block);
    document.querySelector("#previous-block").disabled = !reviewBlocks.some(item => Number(item.dataset.index) < Number(block.dataset.index));
    document.querySelector("#next-block").disabled = !reviewBlocks.some(item => Number(item.dataset.index) > Number(block.dataset.index));
    document.querySelector("#review-position").textContent = position >= 0 ? `Review item ${position + 1} of ${reviewBlocks.length}` : "Surrounding conversation";
    for (const button of document.querySelectorAll("[data-forward]")) {
      button.disabled = document.getElementById(button.dataset.forward).disabled;
    }
  }

  function fromHash() {
    select(byId.get(location.hash.slice(1)) || reviewBlocks[0] || blocks[0]);
  }

  async function play(context = false, full = false) {
    const token = ++request;
    audio.pause();
    stopAt = null;
    status.textContent = "Loading audio…";
    try {
      if (!audio.readyState) {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => finish(new Error("Audio took too long to load.")), 15000);
          const ready = () => finish();
          const failed = () => finish(new Error("Audio could not be loaded."));
          function finish(error) {
            clearTimeout(timer);
            audio.removeEventListener("loadedmetadata", ready);
            audio.removeEventListener("error", failed);
            error ? reject(error) : resolve();
          }
          audio.addEventListener("loadedmetadata", ready);
          audio.addEventListener("error", failed);
          audio.load();
        });
      }
      if (token !== request) return;
      const start = full ? 0 : Math.max(0, Number(selected.dataset.start) - (context ? 6 : 0));
      if (start > 0 && (!audio.seekable.length || audio.seekable.end(audio.seekable.length - 1) < start)) {
        if (!localAudio) {
          localAudio = fetch(audio.currentSrc, {cache: "no-store", signal: AbortSignal.timeout(60000)}).then(response => {
            if (!response.ok) throw new Error("Audio download failed.");
            return response.blob();
          }).then(blob => {
            audioUrl = URL.createObjectURL(blob);
            return new Promise((resolve, reject) => {
              audio.addEventListener("loadedmetadata", resolve, {once: true});
              audio.addEventListener("error", reject, {once: true});
              audio.src = audioUrl;
              audio.load();
            });
          }).catch(error => { localAudio = null; throw error; });
        }
        await localAudio;
        if (token !== request) return;
      }
      stopAt = full ? null : Math.min(audio.duration, Number(selected.dataset.end) + (context ? 6 : 0));
      audio.currentTime = start;
      await audio.play();
      if (token === request) status.textContent = full ? "Playing the full call." : `Playing ${selected.dataset.noteId}${context ? " with surrounding conversation" : ""}.`;
    } catch (error) {
      if (token === request) {
        stopAt = null;
        status.textContent = "Could not start playback. Try the audio player's play button or download the audio.";
      }
    }
  }

  document.querySelector("#play-block").addEventListener("click", () => play());
  document.querySelector("#pause-audio").addEventListener("click", () => { ++request; audio.pause(); status.textContent = "Paused."; });
  document.querySelector("#play-context").addEventListener("click", () => play(true));
  document.querySelector("#play-call").addEventListener("click", () => play(false, true));
  document.querySelector("#speed").addEventListener("change", event => { audio.playbackRate = Number(event.target.value); });
  audio.addEventListener("timeupdate", () => {
    if (stopAt !== null && audio.currentTime >= stopAt) {
      audio.pause();
      stopAt = null;
      status.textContent = "Excerpt finished. Play it again or choose another block.";
    }
  });
  audio.addEventListener("error", () => { status.textContent = "Audio could not be loaded. Try downloading the audio."; });
  audio.addEventListener("ended", () => { stopAt = null; status.textContent = "Call finished."; });
  document.querySelector("#previous-block").addEventListener("click", () => {
    const previous = reviewBlocks.filter(block => Number(block.dataset.index) < Number(selected.dataset.index)).at(-1);
    if (previous) location.hash = previous.id;
  });
  document.querySelector("#next-block").addEventListener("click", () => {
    const next = reviewBlocks.find(block => Number(block.dataset.index) > Number(selected.dataset.index));
    if (next) location.hash = next.id;
  });
  document.querySelector("#copy-id").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(selected.dataset.noteId);
      status.textContent = "Identifier copied. Paste it into your notes.";
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents(document.querySelector("#selected-id"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Identifier selected. Use Copy to put it in your notes.";
    }
  });
  for (const button of document.querySelectorAll("[data-forward]")) {
    button.addEventListener("click", () => document.getElementById(button.dataset.forward).click());
  }
  window.addEventListener("hashchange", fromHash);
  fromHash();
})();
