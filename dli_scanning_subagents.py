"""
DLI Scanning Sub-Agents
Lightweight pattern detection modules that identify vocabulary and grammar usage
and pass findings to main assessment agents for evaluation.
"""

import re
from typing import List, Dict, Tuple


class VocabularyScanningSub:
    """
    Sub-agent that identifies DLI vocabulary terms in student responses
    and extracts context for quality evaluation by the main Lexical Resource Agent.
    """
    
    def scan(self, response: str, dli_items: List[Dict]) -> Dict:
        """
        Scan response for designated DLI vocabulary items.
        
        Args:
            response: Student's written response
            dli_items: List of DLI vocabulary items with term, type, example
            
        Returns:
            Dictionary containing detected terms with context
        """
        if not dli_items:
            return {
                'detected_count': 0,
                'detected_terms': [],
                'scan_summary': 'No DLI vocabulary list provided'
            }
        
        # Filter for vocabulary items only (exclude grammar structures)
        vocab_items = [item for item in dli_items if item.get('type', '') != 'grammar']
        
        detected_terms = []
        response_lower = response.lower()
        
        # Split response into sentences for context extraction
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for item in vocab_items:
            term = item.get('term', '').strip()
            if not term:
                continue
            
            # Check for word boundary match
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, response_lower):
                # Find the sentence containing this term
                context_sentence = None
                for sentence in sentences:
                    if re.search(pattern, sentence.lower()):
                        context_sentence = sentence
                        break
                
                detected_terms.append({
                    'term': term,
                    'type': item.get('type', 'vocabulary'),
                    'context': context_sentence or term,
                    'dli_example': item.get('example', '')
                })
        
        # Remove duplicates based on term
        unique_terms = []
        seen_terms = set()
        for detected in detected_terms:
            if detected['term'].lower() not in seen_terms:
                unique_terms.append(detected)
                seen_terms.add(detected['term'].lower())
        
        return {
            'detected_count': len(unique_terms),
            'detected_terms': unique_terms,
            'scan_summary': f"Found {len(unique_terms)} designated vocabulary items"
        }


class GrammarScanningSub:
    """
    Sub-agent that identifies attempted usage of DLI grammar structures
    and extracts patterns for evaluation by the main Language Control Agent.
    """
    
    def scan(self, response: str, dli_items: List[Dict]) -> Dict:
        """
        Scan response for attempted usage of designated DLI grammar structures.
        
        Args:
            response: Student's written response
            dli_items: List of DLI items including grammar structures
            
        Returns:
            Dictionary containing detected structure attempts with context
        """
        if not dli_items:
            return {
                'detected_count': 0,
                'structure_attempts': [],
                'scan_summary': 'No DLI grammar list provided'
            }
        
        # Filter for grammar structure items only
        grammar_items = [item for item in dli_items if item.get('type', '') == 'grammar']
        
        if not grammar_items:
            return {
                'detected_count': 0,
                'structure_attempts': [],
                'scan_summary': 'No grammar structures in DLI list'
            }
        
        structure_attempts = []
        
        # Split response into sentences for analysis
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Also get vocabulary items that might indicate grammar usage
        # (e.g., modal verbs, conjunctions, specific verb forms)
        vocab_items = [item for item in dli_items if item.get('type', '') != 'grammar']
        
        # Check for grammar indicator words in the response
        for grammar in grammar_items:
            structure_name = grammar.get('term', '').strip()
            examples = grammar.get('example', '')
            
            # Look for sentences that might demonstrate this structure
            relevant_sentences = self._find_relevant_sentences(
                sentences, structure_name, vocab_items, response.lower()
            )
            
            if relevant_sentences:
                structure_attempts.append({
                    'structure_name': structure_name,
                    'attempted_sentences': relevant_sentences,
                    'dli_examples': examples,
                    'indicator_found': True
                })
        
        return {
            'detected_count': len(structure_attempts),
            'structure_attempts': structure_attempts,
            'scan_summary': f"Detected {len(structure_attempts)} potential grammar structure attempts"
        }
    
    def _find_relevant_sentences(
        self, 
        sentences: List[str], 
        structure_name: str,
        vocab_items: List[Dict],
        response_lower: str
    ) -> List[str]:
        """
        Find sentences that might demonstrate the target grammar structure.
        Uses heuristics based on structure name and associated vocabulary.
        """
        relevant = []
        structure_lower = structure_name.lower()
        
        # Define keyword patterns for common grammar structures
        patterns = {
            'modal': ['could', 'would', 'should', 'may', 'might', 'can', 'must'],
            'conditional': ['if', 'unless', 'provided', 'when'],
            'phrasal verb': ['up', 'down', 'out', 'off', 'away', 'back', 'over'],
            'perfect': ['have', 'has', 'had'],
            'passive': ['was', 'were', 'been', 'being'],
            'reported speech': ['said', 'told', 'asked', 'reported'],
            'comparative': ['more', 'less', 'er than', 'as...as'],
            'superlative': ['most', 'least', 'est'],
            'quantifier': ['few', 'little', 'many', 'much', 'some', 'any']
        }
        
        # Find applicable keywords based on structure name
        applicable_keywords = []
        for category, keywords in patterns.items():
            if category in structure_lower:
                applicable_keywords.extend(keywords)
        
        # Also check vocabulary items associated with this structure
        for item in vocab_items:
            item_type = item.get('type', '').lower()
            if any(key in structure_lower for key in ['modal', 'conjunction', 'quantifier', 'phrasal']):
                if item_type in ['modal', 'conjunction', 'quantifier', 'phrasal_verb']:
                    applicable_keywords.append(item.get('term', '').lower())
        
        # Find sentences containing these keywords
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower) 
                   for keyword in applicable_keywords):
                if sentence not in relevant:
                    relevant.append(sentence)
                    if len(relevant) >= 3:  # Limit to 3 examples per structure
                        break
        
        return relevant
