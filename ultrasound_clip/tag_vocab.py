TASKS = {
    # task1: diagnosis
    "task1": ["benign", "malignant"],
    # task2-5: descriptors for thyroid ultrasound
    "task2": ["wider", "taller"],
    "task3": ["smooth", "moderately smooth", "irregular"],
    "task4": [
        "top-left", "top-center", "top-right",
        "middle-left", "middle-center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ],
    "task5": ["none", "tiny", "small", "medium", "large"],
}

TASK_FIELD_MAP = {
    "task1": "Diagnosis",
    "task2": "Shape",
    "task3": "Margins",
    "task4": "Position",
    "task5": "Size",
}

DESCRIPTOR_TASK_IDS = [tk for tk in TASKS.keys() if tk != "task1"]

def build_vocabs():
    diagnosis_vocab = {}
    descriptor_vocab = {}
    for idx, tag in enumerate(TASKS["task1"]):
        diagnosis_vocab[tag] = idx
    offset = 0
    for tk in DESCRIPTOR_TASK_IDS:
        for tag in TASKS[tk]:
            key = f"{tk}::{tag}"
            descriptor_vocab[key] = offset
            offset += 1
    return diagnosis_vocab, descriptor_vocab

DIAGNOSIS_VOCAB, DESCRIPTOR_VOCAB = build_vocabs()
UNK_DIAG_ID = len(DIAGNOSIS_VOCAB)
UNK_DESC_ID = len(DESCRIPTOR_VOCAB)