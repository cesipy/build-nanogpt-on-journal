import os
import json
from typing import List, Dict
import tiktoken
from openai import OpenAI

class InstructSampleGenerator:
    def __init__(self, 
                 api_key: str, 
                 model: str = "gpt-4o-mini",
                 max_tokens: int = 300,
                 temperature: float = 0.7):
        """
        Initialize the Instruct Sample Generator
        
        :param api_key: OpenAI API Key
        :param model: OpenAI model to use
        :param max_tokens: Maximum tokens per response
        :param temperature: Sampling temperature
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.encoding = tiktoken.encoding_for_model(model)

    def generate_samples(self, 
                         journal_text: str, 
                         num_samples: int = 100, 
                         output_file: str = 'custom_instruct_samples.json') -> List[Dict]:
        """
        Generate instruct samples from journal text
        
        :param journal_text: Full journal text
        :param num_samples: Number of samples to generate
        :param output_file: File to save generated samples
        :return: List of generated samples
        """
        # Prompt template for sample generation
        system_prompt = """Du bist ein KI-Assistent, der Instruktions-Beispiele aus einem Tagebuch generiert. 
        Generiere deutsche Instruktions-Antwort-Paare, die:
        - Authentisch den Tagebuchinhalt reflektieren
        - Eine natürliche Gesprächssprache verwenden
        - Persönliche Erfahrungen und Reflexionen einfangen
        - Als JSON-Struktur mit 'human' und 'gpt' Feldern formatiert sind
        Beispiel: 
        [
            {
                "id": "23712",
                "conversations": [
                {
                    "from": "human",
                    "value": "Schlagen Sie einen Slogan für eine Recycling-Kampagne vor.\n"
                },
                {
                    "from": "gpt",
                    "value": "1. \"Reduziere, wiederverwende, recycel: Zusammen für eine grünere Zukunft.\"\n2. \"Recycle heute für eine bessere Zukunft.\"\n3. \"Mache aus deinem Müll einen Schatz - Recycle!\"\n4. \"Recycle für den Lebenszyklus.\"\n5. \"Ressourcen sparen, mehr recyceln.\""
                }
                ]
            }
        ]
        """

        # Truncate journal text if too long
        max_context_tokens = 8000
        truncated_text = self.truncate_text(journal_text, max_context_tokens)

        samples = []
        for i in range(num_samples):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generiere ein einzigartiges Instruktions-Beispiel basierend auf diesem Tagebuchtext. Sei kreativ und persönlich, aber kurze antworten:\n\n{truncated_text}"}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                # Extract and parse the conversation content from the response
                response_content = response.choices[0].message.content
                try:
                    conversation_data = json.loads(response_content)
                    # Keep the generated ID from the conversation or create a new one
                    conversation_data['id'] = f'j24{str(i+100).zfill(3)}'
                    samples.append(conversation_data)
                    # Save to file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, ensure_ascii=False, indent=2)
                except json.JSONDecodeError as e:
                    print(f"Error parsing response content: {e}")
                    continue

                print(f"Generated sample {i+1}/{num_samples}")

            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")
                break

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        

        return samples

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified number of tokens
        
        :param text: Input text
        :param max_tokens: Maximum number of tokens
        :return: Truncated text
        """
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

def main():
    # Replace with your actual OpenAI API key
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Read journal text
    with open('journals.txt', 'r', encoding='utf-8') as f:
        journal_text = f.read()

    # Initialize generator
    generator = InstructSampleGenerator(api_key=API_KEY)
    
    # Generate samples
    samples = generator.generate_samples(journal_text)
    
    print(f"Generated {len(samples)} instruct samples.")

if __name__ == "__main__":
    main()