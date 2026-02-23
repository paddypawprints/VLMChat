/**
 * Few-shot examples for Groq NLP parser
 * These examples train the model to parse natural language queries into structured search terms
 */

export const fewShotExamples = [
  {
    role: "user" as const,
    content: "Young man with a green hoodie and backpack riding a bicycle"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "young man with a green hoodie and backpack riding a bicycle",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Age18-60",\n            "Backpack",\n            "LongSleeve"\n          ],\n          "colors": {\n            "LongSleeve": [0, 128, 0]\n          }\n        },\n        {\n          "category": "bicycle",\n          "colors": {}\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 9,\n    "requires_vlm": false,\n    "vlm_reasoning": "The query can be fully captured with category detection (person, bicycle) and person attributes (Age18-60, Backpack, LongSleeve with green color). No fine-grained details beyond the bitmask."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "women with a green skirt with a blue handbag\n"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "women with a green skirt with a blue handbag",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Female",\n            "Skirt&Dress"\n          ],\n          "colors": {\n            "Skirt&Dress": [0, 128, 0]\n          }\n        },\n        {\n          "category": "handbag",\n          "colors": {\n            "handbag": [0, 0, 255]\n          }\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 9,\n    "requires_vlm": false,\n    "vlm_reasoning": "The query can be captured with person detection (Female, Skirt&Dress with green color) and handbag detection (with blue color). All elements are in the bitmask."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "man with a SF 49ers jersey and a hat\n"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "man with a SF 49ers jersey and a hat",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Age18-60",\n            "Hat",\n            "UpperLogo"\n          ],\n          "colors": {\n            "UpperLogo": [170, 0, 0]\n          }\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 7,\n    "requires_vlm": true,\n    "vlm_reasoning": "While we can detect a person with Hat and UpperLogo attributes with the team color (SF 49ers cardinal red), distinguishing the specific team logo requires VLM verification."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "man with a Chicago bulls jersey and a hat"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "man with a Chicago bulls jersey and a hat",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Age18-60",\n            "Hat",\n            "UpperLogo"\n          ],\n          "colors": {\n            "UpperLogo": [200, 0, 0]\n          }\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 7,\n    "requires_vlm": true,\n    "vlm_reasoning": "While we can detect a person with Hat and UpperLogo with red color, verifying the specific Chicago Bulls logo requires VLM."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "man in a thunder jersey playing basket ball"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "man in a thunder jersey playing basketball",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Age18-60",\n            "UpperLogo"\n          ],\n          "colors": {\n            "UpperLogo": [0, 122, 193]\n          }\n        },\n        {\n          "category": "sports ball",\n          "colors": {}\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 8,\n    "requires_vlm": false,\n    "vlm_reasoning": "The request only needs a person wearing a Thunder jersey (captured via UpperLogo with the team\'s primary color) and a basketball (sports ball). No fine‑grained or abstract details are present."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "student not wearing a backpack"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "student not wearing a backpack",\n      "objects": [\n        {\n          "category": "person",\n          "attributes": [\n            "Age18-60"\n          ],\n          "negative_attributes": [\n            "Backpack"\n          ],\n          "colors": {}\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 6,\n    "requires_vlm": true,\n    "vlm_reasoning": "The query specifies a negative condition (\'not wearing a backpack\') which cannot be directly encoded with the positive attribute set. A vision‑language model is needed to filter out persons who have the Backpack attribute."\n  }\n}'
  },
  {
    role: "user" as const,
    content: "Dalmatian dog\n"
  },
  {
    role: "assistant" as const,
    content: '{\n  "search_terms": [\n    {\n      "search_term": "dalmatian dog",\n      "objects": [\n        {\n          "category": "dog",\n          "attributes": [],\n          "negative_attributes": [],\n          "colors": {\n            "dog": [255, 255, 255]\n          }\n        }\n      ]\n    }\n  ],\n  "strategy": {\n    "completeness_score": 6,\n    "requires_vlm": true,\n    "vlm_reasoning": "The request specifies a Dalmatian, a fine‑grained breed characterized by a white coat with black spots. Breed information and spot pattern cannot be fully expressed with the COCO category/attribute bitmask, so a vision‑language model is needed to disambiguate the specific Dalmatian appearance."\n  }\n}'
  }
];
