#!/usr/bin/env python3
"""
Fast Hugging Face model collection - optimized version
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class FastHFCollector:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.session = requests.Session()
        self.output_dir = Path("data/raw/huggingface")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_models_batch(self, library, limit=500):
        """Fetch a large batch of models at once"""
        params = {
            "limit": limit,
            "sort": "downloads",
            "direction": -1,
            "full": True,
            "library": library
        }

        try:
            response = self.session.get(f"{self.base_url}/models", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {library}: {e}")
            return []

    def fetch_model_details(self, model_id):
        """Fetch model details and card in parallel"""
        result = {"id": model_id}

        # Get model info
        try:
            response = self.session.get(f"{self.base_url}/models/{model_id}", timeout=10)
            if response.status_code == 200:
                result["info"] = response.json()
        except:
            pass

        # Get model card
        try:
            card_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = self.session.get(card_url, timeout=5)
            if response.status_code == 200:
                result["card"] = response.text[:50000]  # Limit card size
        except:
            pass

        return result

    def process_library(self, library, num_models=200):
        """Process all models from a library"""
        print(f"\n📚 Fetching {library} models...")

        # Fetch all models at once
        models = self.fetch_models_batch(library, limit=num_models)
        if not models:
            return []

        print(f"  Found {len(models)} models")

        results = []
        # Process in parallel with thread pool
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}

            for model in models[:num_models]:
                model_id = model.get("modelId", model.get("id"))
                if model_id:
                    future = executor.submit(self.fetch_model_details, model_id)
                    futures[future] = model

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {library}"):
                try:
                    model = futures[future]
                    details = future.result(timeout=10)

                    model_data = {
                        "id": details["id"],
                        "library": library,
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "tags": model.get("tags", []),
                        "lastModified": model.get("lastModified"),
                    }

                    if "info" in details:
                        info = details["info"]
                        model_data.update({
                            "modelSize": info.get("safetensors", {}) if isinstance(info.get("safetensors"), dict) else {},
                            "datasets": info.get("datasets", []),
                            "metrics": info.get("metrics", []),
                            "co2_eq_emissions": info.get("cardData", {}).get("co2_eq_emissions") if info.get("cardData") else None,
                        })

                    if "card" in details:
                        model_data["has_card"] = True
                        # Save card separately
                        card_path = self.output_dir / f"{details['id'].replace('/', '_')}_README.md"
                        with open(card_path, 'w', encoding='utf-8') as f:
                            f.write(details["card"])

                    results.append(model_data)

                except Exception as e:
                    continue

        return results

def main():
    print("⚡ Fast Hugging Face Model Collection")
    print("="*50)

    collector = FastHFCollector()

    # Priority libraries for carbon emission analysis
    libraries = [
        ("transformers", 300),      # NLP models - get more
        ("diffusers", 200),        # Image generation
        ("timm", 150),             # Computer vision
        ("sentence-transformers", 100),  # Embeddings
        ("speechbrain", 50),       # Speech
        ("asteroid", 50),          # Audio
    ]

    all_models = []
    stats = {}

    for library, num_models in libraries:
        models = collector.process_library(library, num_models)
        all_models.extend(models)
        stats[library] = len(models)
        print(f"  ✓ Collected {len(models)} {library} models")

        # Save intermediate results
        with open(f"data/raw/{library}_models.json", 'w') as f:
            json.dump(models, f, indent=2)

    # Save complete dataset
    complete_data = {
        "collection_date": datetime.now().isoformat(),
        "total_models": len(all_models),
        "models_by_library": stats,
        "models_with_carbon_info": sum(1 for m in all_models if m.get("co2_eq_emissions")),
        "models_with_cards": sum(1 for m in all_models if m.get("has_card")),
        "models": all_models
    }

    output_path = Path("data/raw/huggingface_complete.json")
    with open(output_path, 'w') as f:
        json.dump(complete_data, f, indent=2)

    print("\n" + "="*50)
    print("COLLECTION COMPLETE")
    print("="*50)
    print(f"Total models: {len(all_models)}")
    print(f"Models with carbon info: {complete_data['models_with_carbon_info']}")
    print(f"Models with cards: {complete_data['models_with_cards']}")
    print("\nBreakdown by library:")
    for lib, count in stats.items():
        print(f"  {lib}: {count}")
    print(f"\n✅ Data saved to: {output_path}")

if __name__ == "__main__":
    main()