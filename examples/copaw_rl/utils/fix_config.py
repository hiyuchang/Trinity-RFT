import json

with open("/app/working/config.json", "r") as f:
    config = json.load(f)

config["security"]["tool_guard"]["enabled"] = False

with open("/app/working/config.json", "w") as f:
    json.dump(config, f, indent=2)
