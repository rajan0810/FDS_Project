import pandas as pd
from typing import List, Dict, Any

class PaperRecord:
    def __init__(self, paper_id: str, title: str, abstract: str, paragraphs: List[Dict[str, Any]]):
        self.paper_id = paper_id
        self.title = title
        self.abstract = abstract
        self.paragraphs = paragraphs

class QuestionRecord:
    def __init__(self, paper_id: str, question_id: str, question: str, reference_answers: List[str], evidence: List[str], unanswerable: bool):
        self.paper_id = paper_id
        self.question_id = question_id
        self.question = question
        self.reference_answers = reference_answers
        self.evidence = evidence
        self.unanswerable = unanswerable

def flatten_qasper_paper(paper_data) -> PaperRecord:
    paper_id = paper_data["id"]
    title = paper_data["title"]
    abstract = paper_data["abstract"]
    
    full_text = paper_data["full_text"]
    paragraphs = []
    global_idx = 0
    
    # Handle list of dicts (Raw JSON format)
    if isinstance(full_text, list):
        for sec_idx, sec in enumerate(full_text):
            sec_name = sec.get("section_name", "")
            sec_paragraphs = sec.get("paragraphs", [])
            for para_idx, para_text in enumerate(sec_paragraphs):
                words = para_text.split()
                para_id = f"{paper_id}_S{sec_idx:03d}_P{para_idx:03d}"
                paragraphs.append({
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": sec_name,
                    "section_index": sec_idx,
                    "paragraph_index": para_idx,
                    "global_paragraph_index": global_idx,
                    "paragraph_id": para_id,
                    "text": para_text,
                    "word_count": len(words),
                    "character_count": len(para_text),
                    "token_count": int(len(words) * 1.3)
                })
                global_idx += 1
    # Handle dict of lists (Hugging Face format)
    elif isinstance(full_text, dict) and "section_name" in full_text and "paragraphs" in full_text:
        for sec_idx, (sec_name, sec_paragraphs) in enumerate(zip(full_text["section_name"], full_text["paragraphs"])):
            for para_idx, para_text in enumerate(sec_paragraphs):
                words = para_text.split()
                para_id = f"{paper_id}_S{sec_idx:03d}_P{para_idx:03d}"
                paragraphs.append({
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": sec_name,
                    "section_index": sec_idx,
                    "paragraph_index": para_idx,
                    "global_paragraph_index": global_idx,
                    "paragraph_id": para_id,
                    "text": para_text,
                    "word_count": len(words),
                    "character_count": len(para_text),
                    "token_count": int(len(words) * 1.3)
                })
                global_idx += 1
                
    return PaperRecord(paper_id, title, abstract, paragraphs)

def extract_questions(paper_data) -> List[QuestionRecord]:
    paper_id = paper_data["id"]
    qas = paper_data.get("qas", [])
    
    questions = []
    
    # Helper to parse answers
    def parse_answers(answers_list, q_id, q_text):
        ref_answers = []
        evidences = []
        unanswerable = False
        
        for ans in answers_list.get("answer", []) if isinstance(answers_list, dict) else answers_list:
            # Handle if answers_list is actually a list of answer objects with 'answer' keys
            ans_obj = ans.get("answer", ans) if isinstance(ans, dict) else ans
            if not isinstance(ans_obj, dict):
                continue
                
            if ans_obj.get("unanswerable", False):
                unanswerable = True
            
            # Extract free form answer or extractive
            if "free_form_answer" in ans_obj and ans_obj["free_form_answer"]:
                ref_answers.append(ans_obj["free_form_answer"])
            elif "extractive_spans" in ans_obj and ans_obj["extractive_spans"]:
                ref_answers.extend(ans_obj["extractive_spans"])
            
            if "evidence" in ans_obj:
                evidences.extend(ans_obj["evidence"])
                
        questions.append(QuestionRecord(
            paper_id=paper_id,
            question_id=q_id,
            question=q_text,
            reference_answers=list(set(ref_answers)),
            evidence=list(set(evidences)),
            unanswerable=unanswerable
        ))

    # Handle list of dicts (Raw JSON format)
    if isinstance(qas, list):
        for q_obj in qas:
            q_id = q_obj.get("question_id")
            q_text = q_obj.get("question")
            answers_list = q_obj.get("answers", [])
            parse_answers(answers_list, q_id, q_text)
            
    # Handle dict of lists (Hugging Face format)
    elif isinstance(qas, dict) and "question_id" in qas and "question" in qas and "answers" in qas:
        for q_id, q_text, answers_list in zip(qas["question_id"], qas["question"], qas["answers"]):
            parse_answers(answers_list, q_id, q_text)
            
    return questions
