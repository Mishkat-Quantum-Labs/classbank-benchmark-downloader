"""Quick test: convert a single CHA file to JSON using pylangacq."""
import json
import os
import pylangacq

# Parse the Roth transcript
chat = pylangacq.read_chat("dataset/transcripts/Roth/roth.cha", strict=False)

output = {
    "metadata": {
        "file_paths": chat.file_paths,
        "languages": chat.languages(),
        "participants": [str(p) for p in chat.participants()],
        "headers": [str(h) for h in chat.headers()],
    },
    "utterances": []
}

for utt in chat.utterances():
    utterance_data = {
        "participant": utt.participant,
        "tiers": utt.tiers,
        "time_marks": utt.time_marks,
    }
    # Add token-level detail if available
    if utt.tokens:
        utterance_data["tokens"] = [
            {
                "word": tok.word,
                "mor": tok.mor,
                "gra": tok.gra,
            }
            for tok in utt.tokens
        ]
    output["utterances"].append(utterance_data)

# Write output
out_path = "dataset/parsed/roth_sample.json"
os.makedirs("dataset/parsed", exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

print(f"Written {len(output['utterances'])} utterances to {out_path}")
print(f"Output: {out_path}")
