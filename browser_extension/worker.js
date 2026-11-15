import {blockedHost, ruleChange, validDomain} from "./rules.mjs";

const endpoint = "http://127.0.0.1:47843/v1/sites";
const blockedPage = chrome.runtime.getURL("blocked.html");
const alarmName = "check-planner";
let domains = [];
let applied = null;
let sequence = 0;
let pollRunning = false;
let controller;
let queued = Promise.resolve();

function serialize(job) {
  const next = queued.then(job, job);
  queued = next.catch(() => {});
  return next;
}

async function examine(id) {
  let tab;
  try { tab = await chrome.tabs.get(id); } catch { return; }
  const key = "held-" + id;
  const url = tab.pendingUrl || tab.url;
  const host = blockedHost(url, domains);
  if (host) {
    await chrome.storage.session.set({[key]: url});
    try { await chrome.tabs.update(id, {url: blockedPage}); } catch { await chrome.storage.session.remove(key); }
  } else if (url === blockedPage) {
    const original = (await chrome.storage.session.get(key))[key];
    if (!original && domains.length === 0) {
      // The browser may restore a parked tab after restarting, when session
      // storage has expired. Recover its previous history entry if it exists.
      await chrome.tabs.goBack(id).catch(() => {});
    } else if (original && !blockedHost(original, domains)) {
      await chrome.tabs.update(id, {url: original}).catch(() => {});
      await chrome.storage.session.remove(key);
    }
  } else {
    await chrome.storage.session.remove(key);
  }
}

async function apply(sites, revision, status) {
  if (revision !== applied) {
    await chrome.declarativeNetRequest.updateSessionRules(ruleChange(sites));
    domains = sites;
    applied = revision;
    // Before acknowledging a change, cover already displayed pages and release
    // each parked tab whose user hasn't navigated away from our local page.
    for (const tab of await chrome.tabs.query({})) {
      if (tab.id >= 0) await examine(tab.id);
    }
  }
  await chrome.storage.session.set({guardStatus: {message: status, domains}});
}

async function browserID() {
  let id = (await chrome.storage.local.get("browserID")).browserID;
  if (!id) {
    id = crypto.randomUUID();
    await chrome.storage.local.set({browserID: id});
  }
  return id;
}

async function poll() {
  if (pollRunning) return;
  pollRunning = true;
  const version = sequence;
  let retry = false;
  try {
    const token = (await chrome.storage.local.get("pairToken")).pairToken;
    if (!token) {
      await serialize(() => apply([], "", "Unpaired"));
      return;
    }
    const id = await browserID();
    let after = ""; // Always get a fresh policy when the worker wakes.
    while (version === sequence) {
      controller = new AbortController();
      const deadline = setTimeout(() => controller?.abort(), after ? 22000 : 5000);
      let policy;
      try {
        const res = await fetch(endpoint + "?after=" + encodeURIComponent(after), {
          headers: {Authorization: "Bearer " + token, "X-Planner-Browser": id},
          cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal
        });
        if (!res.ok) throw new Error("Planner unavailable or pairing rejected.");
        policy = await res.json();
        if (!/^[0-9a-f]{64}$/.test(policy.revision) || !Array.isArray(policy.domains)) throw new Error("Invalid policy.");
      } finally {
        clearTimeout(deadline);
      }
      if (version !== sequence) break;
      const valid = policy.domains.filter(validDomain);
      const status = valid.length === policy.domains.length
        ? (valid.length ? "Blocking configured sites" : "Connected · no sites blocked")
        : "Skipped an unsupported site";
      await serialize(() => version === sequence && apply(valid, policy.revision, status));
      after = policy.revision;
    }
  } catch {
    if (version === sequence) {
      await serialize(() => apply([], "", "Disconnected · sites released"));
      retry = true;
    }
  } finally {
    pollRunning = false;
    controller = undefined;
    if (version !== sequence) void poll();
    else if (retry) setTimeout(poll, 5000);
  }
}

function changed() {
  sequence++;
  controller?.abort();
  void poll();
}
async function wake() {
  const {pairToken} = await chrome.storage.local.get("pairToken");
  if (pairToken) {
    await chrome.alarms.create(alarmName, {periodInMinutes: 0.5});
  } else {
    await chrome.alarms.clear(alarmName);
  }
  void poll();
}

chrome.alarms.onAlarm.addListener(a => { if (a.name === alarmName) void poll(); });
chrome.runtime.onStartup.addListener(wake);
chrome.runtime.onInstalled.addListener(wake);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.pairToken) {
    changed();
    void wake();
  }
});
chrome.tabs.onActivated.addListener(info => { if (applied !== null) void serialize(() => examine(info.tabId)); });
chrome.tabs.onUpdated.addListener((id, change) => {
  if (applied !== null && (change.url || change.status === "complete")) void serialize(() => examine(id));
});
chrome.tabs.onRemoved.addListener(id => { void serialize(() => chrome.storage.session.remove("held-" + id)); });
chrome.runtime.onMessage.addListener((msg, sender) => {
  if (sender.id === chrome.runtime.id && msg?.type === "sync") changed();
});
wake();
