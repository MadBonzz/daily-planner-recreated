(async () => {
  const tab = await chrome.tabs.getCurrent();
  const key = "held-" + tab.id;
  const url = (await chrome.storage.session.get(key))[key];
  if (url) document.querySelector("#site").textContent = new URL(url).hostname;
  else document.querySelector("#site").textContent = "The previous address is no longer in this browser session. Use Back or enter a new address.";
})();
