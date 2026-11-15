const status = document.querySelector("#status");
async function show() {
  const token = (await chrome.storage.local.get("pairToken")).pairToken;
  const state = (await chrome.storage.session.get("guardStatus")).guardStatus;
  status.textContent = state?.message || (token ? "Connecting…" : "Unpaired");
  document.querySelector("#disconnect").hidden = !token;
  document.querySelector("#sites").replaceChildren(...(state?.domains || []).map(d => {
    const row = document.createElement("li"); row.textContent = d; return row;
  }));
}
document.querySelector("form").addEventListener("submit", async e => {
  e.preventDefault();
  const input = document.querySelector("#token");
  const token = input.value.trim();
  if (!token) return;
  await chrome.storage.local.set({pairToken:token});
  input.value = ""; status.textContent = "Checking pairing…";
  await chrome.runtime.sendMessage({type:"sync"}).catch(() => {});
});
document.querySelector("#disconnect").addEventListener("click", async () => {
  await chrome.storage.local.remove("pairToken");
  await chrome.runtime.sendMessage({type:"sync"}).catch(() => {});
  await show();
});
chrome.storage.onChanged.addListener(show);
show();
