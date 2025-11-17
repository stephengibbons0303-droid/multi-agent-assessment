"""
Multi-Agent Assessment System
Handles assessment of written responses using GPT-4 agents
"""

import openai
from typing import Dict, List, Tuple
import time
import re

class AssessmentAgent:
    """Base class for assessment agents"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def call_gpt(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to GPT-4 with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert language assessment specialist calibrated for CEFR A2/B1 level learners."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def extract_score(self, response: str) -> int:
        """Extract numerical score from agent response"""
        match = re.search(r'SCORE:\s*(\d+)', response)
        if match:
            return int(match.group(1))
        # Fallback: look for any number between 0-100
        numbers = re.findall(r'\b(\d{1,3})\b', response)
        for num in numbers:
            score = int(num)
            if 0 <= score <= 100:
                return score
        return 50  # Default if parsing fails


class LanguageControlAgent(AssessmentAgent):
    """Assesses grammar, syntax, and language accuracy (20% weight)"""
    
    def assess(self, response: str, feedback_level: str) -> Dict:
        prompt = f"""Assess the LANGUAGE CONTROL of this A2/B1 level written response.

Response: "{response}"

Evaluate based on:
- Grammar accuracy
- Sentence structure variety
- Use of tenses
- Subject-verb agreement
- Word order

Provide a score from 0-100 where:
- 0-40: Many basic errors, limited sentence structures
- 41-60: Some errors but generally comprehensible, simple sentences
- 61-80: Minor errors, good variety of structures (A2-B1 appropriate)
- 81-100: Very few errors, excellent control for this level

Format your response as:
SCORE: [number]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(prompt)
        score = self.extract_score(result)
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        
        return {
            'criterion': 'Language Control',
            'score': score,
            'feedback': feedback,
            'raw_response': result
        }


class CoherenceAgent(AssessmentAgent):
    """Assesses coherence and cohesion (20% weight)"""
    
    def assess(self, response: str, feedback_level: str) -> Dict:
        prompt = f"""Assess the COHERENCE AND COHESION of this A2/B1 level written response.

Response: "{response}"

Evaluate based on:
- Logical flow of ideas
- Use of linking words (but, and, because, however, etc.)
- Paragraph organization
- Topic consistency
- Clear progression of thoughts

Provide a score from 0-100 where:
- 0-40: Ideas disconnected, no clear linking
- 41-60: Basic linking, some logical flow
- 61-80: Good use of cohesive devices, clear organization
- 81-100: Excellent coherence with sophisticated linking for this level

Format your response as:
SCORE: [number]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(prompt)
        score = self.extract_score(result)
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        
        return {
            'criterion': 'Coherence and Cohesion',
            'score': score,
            'feedback': feedback,
            'raw_response': result
        }


class LexicalResourceAgent(AssessmentAgent):
    """Assesses vocabulary range and accuracy (20% weight)"""
    
    def assess(self, response: str, feedback_level: str, dli_items: List[str] = None, 
               dli_mode: str = None) -> Dict:
        prompt = f"""Assess the LEXICAL RESOURCE (vocabulary) of this A2/B1 level written response.

Response: "{response}"

Evaluate based on:
- Range of vocabulary
- Appropriate word choice
- Collocations
- Topic-specific vocabulary
- Repetition vs. variety

Provide a score from 0-100 where:
- 0-40: Very limited vocabulary, frequent repetition
- 41-60: Adequate basic vocabulary for A2 level
- 61-80: Good range appropriate for B1 level
- 81-100: Wide vocabulary range with precise usage

Format your response as:
SCORE: [number]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(prompt)
        score = self.extract_score(result)
        
        # Apply DLI boost if applicable
        dli_count = 0
        if dli_items and dli_mode:
            dli_count = self._count_dli_usage(response, dli_items)
            boost_per_item = 3 if dli_mode == 'Closed DLI' else 2
            dli_boost = min(dli_count * boost_per_item, 15)  # Max +15
            score = min(score + dli_boost, 100)
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
            
            if dli_count > 0:
                feedback += f"\n[DLI Items Used: {dli_count}, Boost Applied: +{min(dli_count * (3 if dli_mode == 'Closed DLI' else 2), 15)}]"
        
        return {
            'criterion': 'Lexical Resource',
            'score': score,
            'feedback': feedback,
            'dli_items_used': dli_count,
            'raw_response': result
        }
    
    def _count_dli_usage(self, response: str, dli_items: List[str]) -> int:
        """Count how many DLI vocabulary items are used in the response"""
        response_lower = response.lower()
        count = 0
        for item in dli_items:
            # Simple word boundary check
            if re.search(r'\b' + re.escape(item.lower()) + r'\b', response_lower):
                count += 1
        return count


class TaskAchievementAgent(AssessmentAgent):
    """Assesses how well the response addresses the prompt (40% weight)"""
    
    def assess(self, response: str, prompt: str, feedback_level: str) -> Dict:
        assessment_prompt = f"""Assess the TASK ACHIEVEMENT of this A2/B1 level written response.

Prompt: "{prompt}"
Response: "{response}"

Evaluate based on:
- Does it answer the question/prompt?
- Completeness of response
- Relevance to topic
- Development of ideas
- Appropriate length

Provide a score from 0-100 where:
- 0-40: Minimal response, doesn't address prompt
- 41-60: Partially addresses prompt, limited development
- 61-80: Fully addresses prompt with adequate development
- 81-100: Comprehensive response with well-developed ideas

Format your response as:
SCORE: [number]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            assessment_prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(assessment_prompt)
        score = self.extract_score(result)
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        
        return {
            'criterion': 'Task Achievement',
            'score': score,
            'feedback': feedback,
            'raw_response': result
        }


class VerifierAgent(AssessmentAgent):
    """Verifies assessment consistency and flags anomalies (Option B verification)"""
    
    def verify(self, response: str, agent_scores: Dict[str, int], 
               feedback_level: str) -> Dict:
        """
        Verify agent scores for anomalies:
        - Large score spreads
        - Absolute outliers
        - Response quality vs scores
        """
        
        scores = list(agent_scores.values())
        score_spread = max(scores) - min(scores)
        avg_score = sum(scores) / len(scores)
        
        # Check for anomalies
        anomalies = []
        
        # 1. Score spread check (>25 points)
        if score_spread > 25:
            anomalies.append(f"Large score spread detected: {score_spread} points")
        
        # 2. Individual outliers (>30 points from average)
        for criterion, score in agent_scores.items():
            if abs(score - avg_score) > 30:
                anomalies.append(f"{criterion} score ({score}) significantly differs from average ({avg_score:.1f})")
        
        # 3. Response characteristics check
        word_count = len(response.split())
        if word_count < 30 and avg_score > 70:
            anomalies.append(f"High score ({avg_score:.1f}) for short response ({word_count} words)")
        elif word_count > 200 and avg_score < 40:
            anomalies.append(f"Low score ({avg_score:.1f}) for lengthy response ({word_count} words)")
        
        # If anomalies detected, get verifier's independent assessment
        needs_review = len(anomalies) > 0
        verifier_score = None
        verifier_feedback = ""
        
        if needs_review:
            prompt = f"""As a verification agent, independently assess this A2/B1 response:

Response: "{response}"

Previous agent scores were:
{', '.join([f'{k}: {v}' for k, v in agent_scores.items()])}

Anomalies detected:
{chr(10).join(['- ' + a for a in anomalies])}

Provide your independent overall assessment (0-100) considering:
- Does the score seem fair for the response quality?
- Are there obvious issues the agents might have missed?
- What would be a fair composite score?

Format: SCORE: [number]
REASONING: [brief explanation]
"""
            result = self.call_gpt(prompt)
            verifier_score = self.extract_score(result)
            
            reasoning_match = re.search(r'REASONING:\s*(.+)', result, re.DOTALL)
            if reasoning_match:
                verifier_feedback = reasoning_match.group(1).strip()
        
        return {
            'anomalies_detected': needs_review,
            'anomalies': anomalies,
            'verifier_score': verifier_score,
            'verifier_feedback': verifier_feedback,
            'score_spread': score_spread,
            'original_scores': agent_scores
        }


class DLIAgent(AssessmentAgent):
    """Special agent for assessing DLI vocabulary/grammar usage"""
    
    def assess_dli_usage(self, response: str, dli_items: List[Dict], 
                        dli_mode: str) -> Dict:
        """
        Assess correct usage of DLI vocabulary and grammar structures
        dli_items: List of dicts with keys: term, type, difficulty, example
        """
        
        items_text = "\n".join([f"- {item['term']} ({item['type']}): {item.get('example', '')}" 
                                for item in dli_items])
        
        prompt = f"""Analyze this A2/B1 response for correct usage of specific vocabulary and grammar items.

Response: "{response}"

Target items to check for:
{items_text}

For each item found in the response:
1. Confirm it's used correctly in context
2. Check grammar/usage accuracy

Provide:
- List of correctly used items
- List of attempted but incorrectly used items
- Overall DLI compliance assessment

Format:
CORRECT_ITEMS: [comma-separated list]
INCORRECT_ITEMS: [comma-separated list]
ASSESSMENT: [brief assessment of DLI usage]
"""
        
        result = self.call_gpt(prompt)
        
        # Parse results
        correct_match = re.search(r'CORRECT_ITEMS:\s*(.+)', result)
        incorrect_match = re.search(r'INCORRECT_ITEMS:\s*(.+)', result)
        assessment_match = re.search(r'ASSESSMENT:\s*(.+)', result, re.DOTALL)
        
        correct_items = []
        if correct_match:
            correct_items = [item.strip() for item in correct_match.group(1).split(',') if item.strip()]
        
        incorrect_items = []
        if incorrect_match:
            incorrect_items = [item.strip() for item in incorrect_match.group(1).split(',') if item.strip()]
        
        assessment = ""
        if assessment_match:
            assessment = assessment_match.group(1).strip()
        
        return {
            'correct_items': correct_items,
            'incorrect_items': incorrect_items,
            'correct_count': len(correct_items),
            'total_target_items': len(dli_items),
            'compliance_rate': len(correct_items) / len(dli_items) * 100 if dli_items else 0,
            'assessment': assessment,
            'raw_response': result
        }