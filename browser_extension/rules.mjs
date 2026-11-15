export const RULE = 1;

export function validDomain(value) {
  if (typeof value !== "string" || value.length > 253 || value !== value.toLowerCase()) return false;
  const labels = value.split(".");
  const last = labels.at(-1);
  if (labels.length < 2 || last === "localhost" || /^\d+$/.test(last)) return false;
  return labels.every(label => /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(label));
}

export function blockedHost(url, domains) {
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) return "";
    const host = parsed.hostname.toLowerCase().replace(/\.$/, "");
    return domains.find(d => host === d || host.endsWith("." + d)) || "";
  } catch {
    return "";
  }
}

export function ruleChange(domains) {
  if (!domains.every(validDomain)) throw new Error("Invalid sites in planner response.");
  return {
    removeRuleIds: [RULE],
    addRules: domains.length ? [{
      id: RULE, priority: 1, action: {type: "block"},
      condition: {requestDomains: domains, regexFilter: "^https?://", resourceTypes: ["main_frame"]}
    }] : []
  };
}
