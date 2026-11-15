import test from "node:test";
import assert from "node:assert/strict";
import {blockedHost, ruleChange, validDomain} from "../browser_extension/rules.mjs";

test("a site rule covers a domain and its subdomains at navigation time", () => {
  const selected = ["x.com", "news.bbc.co.uk"];
  const {addRules, removeRuleIds} = ruleChange(selected);
  assert.equal(addRules.length, 1);
  assert.deepEqual(removeRuleIds, [addRules[0].id]);
  assert.deepEqual(addRules[0].condition, {regexFilter: "^https?://", requestDomains: selected, resourceTypes: ["main_frame"]});
  assert.equal(addRules[0].action.type, "block");
  assert.equal(blockedHost("https://mobile.x.com/feed", selected), "x.com");
  for (const url of ["https://x.com.evil.org", "https://suffixx.com", "http://127.0.0.1:8501", "chrome://settings"]) {
    assert.equal(blockedHost(url, selected), "");
  }
  assert.deepEqual(ruleChange([]), {removeRuleIds, addRules: []});
});

test("unsupported rules cannot target the planner or its pairing API", () => {
  for (const raw of ["*", "*.com", "localhost", "127.0.0.1", "a.localhost", "::1", "https://x.com", ""]) {
    assert.equal(validDomain(raw), false, raw);
  }
  for (const raw of ["x.com", "a.example.test", "news.example.edu", "xn--bcher-kva.de"]) {
    assert.equal(validDomain(raw), true, raw);
  }
});

test("worker parks open tabs, updates on a task switch and restores on disconnect", async () => {
  const {mkdtemp, copyFile, rm} = await import("node:fs/promises");
  const {tmpdir} = await import("node:os");
  const {join} = await import("node:path");
  const {pathToFileURL} = await import("node:url");
  const {createHash} = await import("node:crypto");
  const root = await mkdtemp(join(tmpdir(), "plan-focus-worker-"));
  await copyFile(new URL("../browser_extension/worker.js", import.meta.url), join(root, "worker.mjs"));
  await copyFile(new URL("../browser_extension/rules.mjs", import.meta.url), join(root, "rules.mjs"));
  function event() {
    const listeners = [];
    return {addListener: f => listeners.push(f), emit: (...args) => listeners.forEach(f => f(...args))};
  }
  function area(records) {
    return {
      get: async k => ({[k]:records[k]}), set: async obj => Object.assign(records,obj),
      remove: async k => { delete records[k]; }
    };
  }
  const session = {}, local = {pairToken:"test-pair",browserID:"test-client-01"};
  const tabs = new Map([[1,{id:1,url:"https://x.com/feed?a=1"}], [2,{id:2,url:"https://reddit.com/r/one"}],
    [3,{id:3,url:"http://127.0.0.1:8501"}]]);
  const originalFetch = globalThis.fetch;
  let rules, respond;
  globalThis.chrome = {
    runtime:{id:"a".repeat(32),getURL:s=>"chrome-extension://"+"a".repeat(32)+"/"+s,
      onInstalled:event(),onStartup:event(),onMessage:event()},
    storage:{session:area(session),local:area(local),onChanged:event()},
    alarms:{create:async()=>{},clear:async()=>{},onAlarm:event()},
    declarativeNetRequest:{updateSessionRules:async action=>{rules=action.addRules;}},
    tabs:{query:async()=>[...tabs.values()],get:async id=>tabs.get(id),
      update:async(id, props)=>Object.assign(tabs.get(id),props),
      onActivated:event(),onUpdated:event(),onRemoved:event()}
  };
  globalThis.fetch = async (url, options) => {
    assert.equal(new URL(url).hostname,"127.0.0.1");
    assert.equal(options.headers.Authorization,"Bearer test-pair");
    return await new Promise((resolve,reject) => {
      respond = sites => sites ? resolve({ok:true,json:async()=>({
        domains:sites,revision:createHash("sha256").update(JSON.stringify(sites)).digest("hex")})}) : reject(Error("disconnected"));
      options.signal.addEventListener("abort", () => reject(Error("interrupted")), {once:true});
    });
  };
  const until = async f => {
    for(let i=0;i<200 && !f();i++) await new Promise(r=>setTimeout(r,5));
    assert.ok(f());
  };
  try {
    await import(pathToFileURL(join(root,"worker.mjs")));
    await until(()=>respond);
    respond(["x.com"]); await until(()=>session.guardStatus?.domains.includes("x.com"));
    assert.equal(tabs.get(1).url,chrome.runtime.getURL("blocked.html"));
    assert.equal(session["held-1"],"https://x.com/feed?a=1");
    assert.equal(tabs.get(3).url,"http://127.0.0.1:8501");
    respond(["reddit.com"]); await until(()=>session.guardStatus?.domains.includes("reddit.com"));
    assert.equal(tabs.get(1).url,"https://x.com/feed?a=1");
    assert.equal(tabs.get(2).url,chrome.runtime.getURL("blocked.html"));
    respond(null); await until(()=>session.guardStatus?.message.includes("Disconnected"));
    assert.deepEqual(rules,[]);
    assert.equal(tabs.get(2).url,"https://reddit.com/r/one");
    delete local.pairToken;
    chrome.storage.onChanged.emit({pairToken:{}}, "local");
    await until(()=>session.guardStatus?.message==="Unpaired");
  } finally {
    globalThis.fetch=originalFetch;
    await rm(root,{recursive:true,force:true});
  }
});
