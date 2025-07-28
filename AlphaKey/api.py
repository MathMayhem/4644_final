# api.py

import requests
import sys

from config import Config

class KeyboardEnvironment:
    """
    A client class that provides an interface to the external scoring API.
    """
    def __init__(self):
        self.api_url = Config.API_URL

    def get_score_components(self, layout_str: str, weights_dict: dict) -> dict:
        """
        Processes a single layout.
        """
        batch_result = self.get_score_components_batched([(layout_str, weights_dict)])
        return batch_result[0]

    def get_score_components_batched(self, layouts_and_weights: list) -> list:
        """
        Processes a batch of layouts and weights in a single API call.
        """
        request_batch = [
            {"layout": layout, "weights": weights}
            for layout, weights in layouts_and_weights
        ]

        try:
            response = requests.post(self.api_url, json=request_batch, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if not isinstance(response_data, list) or len(response_data) != len(request_batch):
                print("API Error: Batched response is malformed.", file=sys.stderr)
                return [ {k: 0.0 for k in Config.WEIGHT_KEYS} ] * len(layouts_and_weights)

            final_components_batch = []
            for i in range(len(request_batch)):
                request_item = request_batch[i]
                result_item = response_data[i]

                raw_stat_values = result_item.get('stat_values')
                if not raw_stat_values:
                    final_components_batch.append({k: 0.0 for k in Config.WEIGHT_KEYS})
                    continue

                final_components = {
                    key: raw_stat_values.get(key, 0.0) * request_item['weights'].get(key, 0.0)
                    for key in Config.WEIGHT_KEYS
                }
                final_components_batch.append(final_components)

            return final_components_batch

        except requests.exceptions.RequestException as e:
            print(f"API batch request failed: {e}", file=sys.stderr)
            return [ {k: 0.0 for k in Config.WEIGHT_KEYS} ] * len(layouts_and_weights)
