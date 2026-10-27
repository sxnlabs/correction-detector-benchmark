from datetime import UTC, datetime

from cdbench.dataset import Message
from cdbench.labeling.arbitration import ReviewItem
from scripts.telegram_review import answer_to_label, collect_answers, send_batch


def m(i):
    return Message(id=str(i), ts=datetime(2026, 10, 1, tzinfo=UTC), text=f"message {i}",
                   prior_reply=None, llm_eligible=True)


class FakeRouter:
    def __init__(self):
        self.sent = []
        self.answers = {}

    def ask(self, text, *, agent, urgency, options):
        qid = f"q-{len(self.sent)}"
        self.sent.append((qid, text, agent, urgency, tuple(options)))
        return qid

    def find_answer(self, qid):
        return self.answers.get(qid)


def test_send_is_idempotent(tmp_path):
    router = FakeRouter()
    items = [ReviewItem(str(i), "opus_vs_haiku") for i in range(15)]
    messages = {str(i): m(i) for i in range(15)}
    first = send_batch(items, messages, data_dir=tmp_path, router=router, batch_size=10)
    second = send_batch(items, messages, data_dir=tmp_path, router=router, batch_size=10)
    assert len(first) == 10 and second == []  # waits until the first batch is answered
    assert router.sent[0][3] == "urgent" and router.sent[0][4] == ("Correction", "Autre")


def test_next_batch_after_answers(tmp_path):
    router = FakeRouter()
    items = [ReviewItem(str(i), "opus_vs_haiku") for i in range(15)]
    messages = {str(i): m(i) for i in range(15)}
    send_batch(items, messages, data_dir=tmp_path, router=router, batch_size=10)
    for i in range(10):
        router.answers[f"q-{i}"] = {"answer": f"q-{i}:Autre"}
    second = send_batch(items, messages, data_dir=tmp_path, router=router, batch_size=10)
    assert len(second) == 5


def test_answer_mapping():
    assert answer_to_label("q-1:Correction") is True
    assert answer_to_label("q-1:Autre") is False
    assert answer_to_label("correction oui") is True
    assert answer_to_label("autre") is False


def test_free_text_ambiguous_is_unanswered(tmp_path):
    assert answer_to_label("c'est quoi ?") is None
    router = FakeRouter()
    items = [ReviewItem("1", "opus_vs_haiku")]
    send_batch(items, {"1": m(1)}, data_dir=tmp_path, router=router)
    router.answers["q-0"] = {"answer": "je ne sais pas"}
    assert collect_answers(tmp_path, router=router) == {}


def test_next_batch_after_12_hours_even_if_unanswered(tmp_path):
    from datetime import timedelta

    router = FakeRouter()
    items = [ReviewItem(str(i), "opus_vs_haiku") for i in range(15)]
    messages = {str(i): m(i) for i in range(15)}
    t0 = datetime(2026, 10, 27, 8, tzinfo=UTC)
    send_batch(items, messages, data_dir=tmp_path, router=router, now=t0)
    assert send_batch(items, messages, data_dir=tmp_path, router=router,
                      now=t0 + timedelta(hours=11)) == []
    later = send_batch(items, messages, data_dir=tmp_path, router=router,
                       now=t0 + timedelta(hours=12, minutes=1))
    assert len(later) == 5


def test_question_reads_in_order_and_defines_the_answer():
    from scripts.telegram_review import _question_text

    msg = Message(id="1", ts=datetime(2026, 10, 1, tzinfo=UTC), text="mon message",
                  prior_reply="sa réponse", llm_eligible=True)
    text = _question_text(msg, 3, 80)
    assert text.index("sa réponse") < text.index("mon message")  # chronological
    assert "Tu as ensuite écrit" in text
    assert "Correction = reproche, consigne durable" in text
    assert "Autre = demande, question, info" in text
    no_prior = _question_text(m(2), 1, 80)
    assert "Hermès t'avait répondu" not in no_prior and "message 2" in no_prior
