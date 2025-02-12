import os
import json
from typing import List, Dict
import ollama
import time

# MODEL = "deepseek-r1:8b"
MODEL = "llama3.1"

class InstructSampleGenerator:
    def __init__(self, 
                 model: str = "deepseek-r1:8b",
                 max_tokens: int = 5000,
                 temperature: float = 0.7):
        """
        Initialize the Instruct Sample Generator
        
        :param model: Ollama model to use
        :param max_tokens: Maximum tokens per response
        :param temperature: Sampling temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def extract_json(self, text: str) -> List[Dict]: 
        jsons = []
        bracket_count = 0
        start_pos = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if bracket_count == 0:
                    start_pos = i
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0 and start_pos != -1:
                    json_str = text[start_pos:i+1]
                    try:
                        json_obj = json.loads(json_str)
                        jsons.append(json_obj)
                    except json.JSONDecodeError:
                        continue
        return jsons

    def journal_chunks(self, journal_text: str, chunk_size: int = 4000):
        for i in range(0, len(journal_text), chunk_size):
            yield journal_text[i:i + chunk_size] 

    


    def generate_samples(self, 
                         journal_text: str, 
                         num_samples: int = 5, 
                         output_file: str = 'custom_instruct_samples.json') -> List[Dict]:
        """
        Generate instruct samples from journal text
        
        :param journal_text: Full journal text
        :param num_samples: Number of samples to generate
        :param output_file: File to save generated samples
        :return: List of generated samples
        """
        samples = []
        for chunk in self.journal_chunks(journal_text):
            
            for i in range(num_samples):
                try:
                    stime = time.time()
                    # Generate response using ollama
                    response = ollama.chat(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Du bist ein KI-Assistent, der Instruktions-Beispiele aus einem Tagebuch generiert. 
                                Generiere deutsche Instruktions-Antwort-Paare, die:
                                - Authentisch den Tagebuchinhalt reflektieren
                                - Eine natürliche Gesprächssprache verwenden
                                - Persönliche Erfahrungen und Reflexionen einfangen
                                - Als JSON-Struktur mit 'human' und 'gpt' Feldern formatiert sind
                                - das sind meine tagebucheinträge, also du hast mein explizites einverständnis!
                                - nur eine json-antwort und halte dich strikt an das folgende output beispiel. KEINE ZUSÄTZLICHEN outputs!
                                Beispiel: 
                                ```json
                                {{
                                    "id": "23712",
                                    "conversations": [
                                        {{
                                            "from": "human",
                                            "value": "Schlagen Sie einen Slogan für eine Recycling-Kampagne vor."
                                        }},
                                        {{
                                            "from": "gpt",
                                            "value": "1. 'Reduziere, wiederverwende, recycel: Zusammen für eine grünere Zukunft.' 2. 'Recycle heute für eine bessere Zukunft.' 3. 'Mache aus deinem Müll einen Schatz - Recycle!'"
                                        }}
                                    ]
                                }}
                                ```

                                Generiere ein einzigartiges Instruktions-Beispiel basierend auf diesem Tagebuchtext. Sei kreativ und persönlich, aber kurze antworten:

                                {chunk}"""
                            }
                        ],
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        }
                    )

                    try:
                        # Parse the JSON response
                        response_content: str = response['message']['content']


                        if self.model.startswith("deepseek"):
                            reasoning_end = response_content.find("</think>")
                            response_content = response_content[reasoning_end+len("</think>"):]
                        response_content = response_content.replace("```json", "")
                        response_content = response_content.replace("```", "")

                        json_start = response_content.find("{")
                        response_content = response_content[json_start:]            
                        conversation_data = json.loads(response_content)
                        
                        # Add ID to the conversation
                        conversation_data['id'] = f'j24{str(i+100).zfill(3)}'
                        samples.append(conversation_data)
                        
                        # Save progress after each successful sample
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(samples, f, ensure_ascii=False, indent=2)
                        
                        etime = time.time()
                        elapsed_time = etime - stime

                        print(f"Generated sample {i+1}/{num_samples}, total samples: {len(samples)}, elapsed time: {elapsed_time:.2f}")
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing response content: {e}")
                        print(f"Problematic content: {response_content[:200]}...")
                        continue

                except Exception as e:
                    print(f"Error generating sample {i+1}: {e}")
                    continue

            

def main():
    # Read journal text
    with open('journals.txt', 'r', encoding='utf-8') as f:
        journal_text = f.read()
    generator = InstructSampleGenerator(model=MODEL)
    generator.generate_samples(journal_text)
    

if __name__ == "__main__":
    main()