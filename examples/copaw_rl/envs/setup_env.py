import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    rl_server_path = "/app/working.secret/providers/custom/rl-server.json"
    with open(rl_server_path, "r") as f:
        data = json.load(f)
    data["base_url"] = args.base_url
    data["extra_models"][0]["id"] = args.model_path
    with open(rl_server_path, "w") as f:
        json.dump(data, f)

    activate_model_path = "/app/working.secret/providers/active_model.json"
    with open(activate_model_path, "r") as f:
        data = json.load(f)
    data["model"] = args.model_path
    with open(activate_model_path, "w") as f:
        json.dump(data, f)
