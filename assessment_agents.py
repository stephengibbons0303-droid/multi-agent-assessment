"""
Multi-Agent Assessment System with DLI Sub-Agent Integration
Handles assessment of written responses using GPT-4 agents with optional DLI scanning
"""

import openai
from typing import Dict, List, Tuple, Optional
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
    """
    Assesses grammar, syntax, and language accuracy (base 16% weight + up to 4% bonus)
    Integrates with Grammar Scanning Sub-Agent for DLI bonus evaluation
    """
    
    def assess(self, response: str, feedback_level: str, grammar_scan_results: Optional[Dict] = None) -> Dict:
        """
        Assess language control with optional DLI grammar bonus evaluation
        
        Args:
            response: Student's written response
            feedback_level: A, B, C, or D
            grammar_scan_results: Optional findings from grammar scanning sub-agent
        """
        prompt = f"""Assess the LANGUAGE CONTROL (grammar, syntax, mechanics) of this A2/B1 level written response.

Response: "{response}"

Evaluate based on:
- Grammar accuracy (verb tense, word order, agreement)
- Sentence structure variety and correctness
- Punctuation and spelling
- Overall linguistic precision

BASE SCORING (0-16 points):
- 0-4: Frequent errors obscuring meaning
- 5-8: Generally correct but consistent errors
- 9-12: Mostly accurate with varied structures, minor errors
- 13-16: Excellent control with very few slips

Format your base assessment as:
BASE_SCORE: [number 0-16]
"""

        # Add DLI grammar bonus evaluation if scan results provided
        bonus_prompt = ""
        if grammar_scan_results and grammar_scan_results.get('detected_count', 0) > 0:
            bonus_prompt = f"""

DLI GRAMMAR BONUS EVALUATION (0-4 points):
The student's response has been scanned for designated grammar structures from the curriculum.

Detected structure attempts:
"""
            for attempt in grammar_scan_results.get('structure_attempts', []):
                bonus_prompt += f"\nStructure: {attempt['structure_name']}"
                bonus_prompt += f"\nExamples from curriculum: {attempt['dli_examples']}"
                bonus_prompt += f"\nStudent's sentences: {'; '.join(attempt['attempted_sentences'][:2])}"
                bonus_prompt += "\n"
            
            bonus_prompt += """
Evaluate the grammar structure usage and assign BONUS points:
- 0 points: No designated structures attempted
- 1 point: Attempts one structure but incorrectly applied
- 2 points: Uses one structure with limited success, partial understanding
- 3 points: Uses one structure correctly with minor issues
- 4 points: Uses one or more structures correctly and appropriately

BONUS_SCORE: [number 0-4]
BONUS_EXPLANATION: [brief justification]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        prompt += bonus_prompt
        
        result = self.call_gpt(prompt)
        base_score = self._extract_base_score(result)
        bonus_score = self._extract_bonus_score(result) if grammar_scan_results else 0
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=BONUS_SCORE:|$)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        
        bonus_explanation = ""
        if grammar_scan_results and bonus_score > 0:
            bonus_match = re.search(r'BONUS_EXPLANATION:\s*(.+?)$', result, re.DOTALL)
            if bonus_match:
                bonus_explanation = bonus_match.group(1).strip()
        
        return {
            'criterion': 'Language Control',
            'base_score': base_score,
            'bonus_score': bonus_score,
            'total_score': base_score + bonus_score,
            'feedback': feedback,
            'bonus_explanation': bonus_explanation,
            'raw_response': result
        }
    
    def _extract_base_score(self, response: str) -> int:
        """Extract base score (0-16)"""
        match = re.search(r'BASE_SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 16)
        return 8  # Default middle score
    
    def _extract_bonus_score(self, response: str) -> int:
        """Extract bonus score (0-4)"""
        match = re.search(r'BONUS_SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 4)
        return 0


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

Provide a score from 0-20 where:
- 0-5: Ideas disconnected, no clear linking
- 6-10: Basic linking, some logical flow
- 11-15: Good use of cohesive devices, clear organization
- 16-20: Excellent coherence with sophisticated linking for this level

Format your response as:
SCORE: [number 0-20]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(prompt)
        score = self._extract_score_range(result, 0, 20)
        
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
    
    def _extract_score_range(self, response: str, min_score: int, max_score: int) -> int:
        """Extract score within specified range"""
        match = re.search(r'SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, min_score), max_score)
        return (min_score + max_score) // 2  # Default middle score


class LexicalResourceAgent(AssessmentAgent):
    """
    Assesses vocabulary range and accuracy (base 16% weight + up to 4% bonus)
    Integrates with Vocabulary Scanning Sub-Agent for DLI bonus evaluation
    """
    
    def assess(self, response: str, feedback_level: str, vocab_scan_results: Optional[Dict] = None, 
               dli_mode: str = None) -> Dict:
        """
        Assess lexical resource with optional DLI vocabulary bonus evaluation
        
        Args:
            response: Student's written response
            feedback_level: A, B, C, or D
            vocab_scan_results: Optional findings from vocabulary scanning sub-agent
            dli_mode: Closed DLI, Open DLI, or None
        """
        prompt = f"""Assess the LEXICAL RESOURCE (vocabulary) of this A2/B1 level written response.

Response: "{response}"

Evaluate based on:
- Range of vocabulary
- Appropriate word choice
- Collocations
- Topic-specific vocabulary
- Repetition vs. variety

BASE SCORING (0-16 points):
- 0-4: Very limited vocabulary, frequent repetition
- 5-8: Adequate basic vocabulary for A2 level
- 9-12: Good range appropriate for B1 level
- 13-16: Wide vocabulary range with precise usage

Format your base assessment as:
BASE_SCORE: [number 0-16]
"""

        # Add DLI vocabulary bonus evaluation if scan results provided
        bonus_prompt = ""
        if vocab_scan_results and vocab_scan_results.get('detected_count', 0) > 0:
            bonus_prompt = f"""

DLI VOCABULARY BONUS EVALUATION (0-4 points):
The student used {vocab_scan_results['detected_count']} designated vocabulary items from the curriculum:

"""
            for detected in vocab_scan_results.get('detected_terms', [])[:6]:  # Show up to 6
                bonus_prompt += f"- '{detected['term']}' (context: {detected['context']})\n"
            
            bonus_prompt += """
Evaluate whether each vocabulary item is used correctly and naturally in context.
Award 1 bonus point per correctly and appropriately used designated term, maximum 4 points.

BONUS_SCORE: [number 0-4]
BONUS_EXPLANATION: [brief note on which terms earned credit]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        prompt += bonus_prompt
        
        result = self.call_gpt(prompt)
        base_score = self._extract_base_score(result)
        bonus_score = self._extract_bonus_score(result) if vocab_scan_results else 0
        
        feedback = ""
        if feedback_level in ['B', 'C', 'D']:
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=BONUS_SCORE:|$)', result, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
        
        bonus_explanation = ""
        if vocab_scan_results and bonus_score > 0:
            bonus_match = re.search(r'BONUS_EXPLANATION:\s*(.+?)$', result, re.DOTALL)
            if bonus_match:
                bonus_explanation = bonus_match.group(1).strip()
        
        return {
            'criterion': 'Lexical Resource',
            'base_score': base_score,
            'bonus_score': bonus_score,
            'total_score': base_score + bonus_score,
            'dli_items_detected': vocab_scan_results.get('detected_count', 0) if vocab_scan_results else 0,
            'feedback': feedback,
            'bonus_explanation': bonus_explanation,
            'raw_response': result
        }
    
    def _extract_base_score(self, response: str) -> int:
        """Extract base score (0-16)"""
        match = re.search(r'BASE_SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 16)
        return 8  # Default middle score
    
    def _extract_bonus_score(self, response: str) -> int:
        """Extract bonus score (0-4)"""
        match = re.search(r'BONUS_SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 4)
        return 0


class TaskAchievementAgent(AssessmentAgent):
    """Assesses how well the response addresses the prompt (40% weight)"""
    
    def assess(self, response: str, prompt: str, feedback_level: str, word_count: int = None) -> Dict:
        # Calculate word count if not provided
        if word_count is None:
            word_count = len(response.split())
        
        assessment_prompt = f"""Assess the TASK ACHIEVEMENT of this A2/B1 level written response.

Prompt: "{prompt}"
Response: "{response}"
Word Count: {word_count}

Evaluate based on:
- Does it answer the question/prompt?
- Completeness of response
- Relevance to topic
- Development of ideas
- Appropriate length

LENGTH CONSIDERATIONS:
- Responses under 10 words: Severely inadequate, maximum score 10/40
- Responses 10-19 words: Minimal development, maximum score 20/40
- Responses 20-65 words: Appropriate length range, score on content merit
- Responses over 65 words: Evaluate whether excessive length impairs clarity or focus

Provide a score from 0-40 where:
- 0-10: Minimal response, doesn't address prompt, or severely too brief
- 11-20: Partially addresses prompt, limited development, or inadequate length
- 21-30: Fully addresses prompt with adequate development
- 31-40: Comprehensive response with well-developed ideas

Format your response as:
SCORE: [number 0-40]
"""
        
        if feedback_level in ['B', 'C', 'D']:
            assessment_prompt += "\nFEEDBACK: [brief feedback]" if feedback_level == 'B' else "\nFEEDBACK: [detailed paragraph explaining strengths and weaknesses]"
        
        result = self.call_gpt(assessment_prompt)
        score = self._extract_score_range(result, 0, 40)
        
        # Apply hard caps based on word count
        if word_count < 10:
            score = min(score, 10)
        elif word_count < 20:
            score = min(score, 20)
        
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
    
    def _extract_score_range(self, response: str, min_score: int, max_score: int) -> int:
        """Extract score within specified range"""
        match = re.search(r'SCORE:\s*(\d+)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, min_score), max_score)
        return (min_score + max_score) // 2  # Default middle score


class VerifierAgent(AssessmentAgent):
    """Verifies assessment consistency and flags anomalies"""
    
    def verify(self, response: str, agent_scores: Dict[str, float], 
               feedback_level: str) -> Dict:
        """
        Verify agent scores for anomalies
        
        Args:
            response: Student's response text
            agent_scores: Dictionary of criterion names to scores
            feedback_level: A, B, C, or D
        """
        scores = list(agent_scores.values())
        score_spread = max(scores) - min(scores)
        avg_score = sum(scores) / len(scores)
        
        # Check for anomalies
        anomalies = []
        
        # 1. Score spread check (>25 points)
        if score_spread > 25:
            anomalies.append(f"Large score spread detected: {score_spread:.1f} points")
        
        # 2. Individual outliers (>30 points from average)
        for criterion, score in agent_scores.items():
            if abs(score - avg_score) > 30:
                anomalies.append(f"{criterion} score ({score:.1f}) significantly differs from average ({avg_score:.1f})")
        
        # 3. Response characteristics check
        word_count = len(response.split())
        if word_count < 30 and avg_score > 70:
            anomalies.append(f"High score ({avg_score:.1f}) for short response ({word_count} words)")
        elif word_count > 200 and avg_score < 40:
            anomalies.append(f"Low score ({avg_score:.1f}) for lengthy response ({word_count} words)")
        
        needs_review = len(anomalies) > 0
        
        return {
            'anomalies_detected': needs_review,
            'anomalies': anomalies,
            'score_spread': score_spread,
            'original_scores': agent_scores
        }


class DLIAgent(AssessmentAgent):
    """Legacy DLI agent - functionality now handled by scanning sub-agents"""
    
    def assess_dli_usage(self, response: str, dli_items: List[Dict], 
                        dli_mode: str) -> Dict:
        """Legacy method maintained for backward compatibility"""
        return {
            'correct_items': [],
            'incorrect_items': [],
            'correct_count': 0,
            'total_target_items': len(dli_items),
            'compliance_rate': 0,
            'assessment': 'Legacy DLI agent - use scanning sub-agents instead',
            'raw_response': ''
        }
