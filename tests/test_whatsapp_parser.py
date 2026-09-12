import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from parser.whatsapp_parser import WhatsAppParser


def test_basic_message():
    text = "[9/5/26, 7:31:22 PM] You: hey"
    result = WhatsAppParser().parse_text(text)

    assert len(result.messages) == 1
    msg = result.messages[0]
    assert msg.sender == "You"
    assert msg.text == "hey"
    assert msg.message_type == "text"
    assert msg.timestamp == "2026-09-05T19:31:22"
    assert msg.raw_line_start == 1
    assert msg.raw_line_end == 1


def test_multiline_message_stays_one_message():
    text = (
        "[9/3/26, 7:54:09 PM] You: [Forwarded] US work hours glazed lagta hai\n"
        "I know my friend who works in us hours and his life is shit\n"
        "So detached from the world\n"
        "\n"
        "He works remotely from india\n"
        "[9/3/26, 7:54:25 PM] You: ye tech hena"
    )
    result = WhatsAppParser().parse_text(text)

    assert len(result.messages) == 2
    first = result.messages[0]

    assert first.is_forwarded is True
    assert first.text == (
        "[Forwarded] US work hours glazed lagta hai\n"
        "I know my friend who works in us hours and his life is shit\n"
        "So detached from the world\n"
        "\n"
        "He works remotely from india"
    )
    assert first.raw_line_start == 1
    assert first.raw_line_end == 5


def test_forwarded_is_not_a_block():
    text = (
        "[9/3/26, 7:54:07 PM] You: [Forwarded] first\n"
        "[9/3/26, 7:54:07 PM] You: [Forwarded] second\n"
        "[9/3/26, 7:54:08 PM] You: [Forwarded] third\n"
        "continuation"
    )
    result = WhatsAppParser().parse_text(text)

    assert len(result.messages) == 3
    assert [m.is_forwarded for m in result.messages] == [True, True, True]
    assert result.messages[2].text == "[Forwarded] third\ncontinuation"


def test_media_types():
    text = (
        "[6/5/26, 9:53:48 PM] Alice: <GIF omitted>\n"
        "[9/7/26, 6:37:43 PM] Alice: <video omitted>\n"
        "[9/11/26, 11:43:32 AM] You: <document omitted>\n"
        "[9/11/26, 11:21:18 AM] You: <image omitted>\n"
        "[6/19/26, 7:00:34 PM] Alice: <album message>"
    )
    result = WhatsAppParser().parse_text(text)

    assert [m.message_type for m in result.messages] == [
        "gif", "video", "document", "image", "album"
    ]
    assert result.messages[0].is_media is True
    assert result.messages[1].is_media is True
    # A document with a filename is still text from the export's perspective;
    # the classifier is intentionally conservative until a richer media grammar
    # is added.
    assert result.messages[4].is_media is True


def test_document_and_forwarded_document_are_classified_as_document():
    text = (
        "[9/11/26, 2:07:48 PM] Alice: "
        "[Forwarded] <document omitted> SampleDoc.pdf\n"
        "[9/11/26, 2:07:56 PM] Alice: project resume doc"
    )
    result = WhatsAppParser().parse_text(text)

    # This assertion captures a requirement we want in the implementation.
    assert result.messages[0].message_type == "document"
    assert result.messages[0].is_forwarded is True


def test_deleted_and_call():
    text = (
        "[7/29/26, 5:35:34 AM] Alice: This message was deleted\n"
        "[6/17/26, 9:44:18 PM] - [Call]"
    )
    result = WhatsAppParser().parse_text(text)

    assert result.messages[0].message_type == "deleted"
    assert result.messages[0].is_system is False
    assert result.messages[1].message_type == "call"
    assert result.messages[1].is_system is True


def test_colon_inside_message():
    text = "[9/11/26, 11:48:04 AM] Alice: URL: https://example.com?a=1:b"
    result = WhatsAppParser().parse_text(text)

    assert result.messages[0].sender == "Alice"
    assert result.messages[0].text == "URL: https://example.com?a=1:b"


def test_orphan_lines_are_not_silently_lost():
    text = "random preamble\n[9/5/26, 7:31:22 PM] You: hey"
    result = WhatsAppParser().parse_text(text)

    assert len(result.messages) == 1
    assert len(result.issues) == 1
    assert result.issues[0].line_number == 1


def test_malformed_header_like_line_is_treated_as_continuation():
    text = (
        "[9/5/26, 7:31:22 PM] You: hello\n"
        "[not a real whatsapp header]\n"
        "still part of message\n"
        "[9/5/26, 7:32:22 PM] You: next"
    )
    result = WhatsAppParser().parse_text(text)

    assert len(result.messages) == 2
    assert result.messages[0].text == (
        "hello\n[not a real whatsapp header]\nstill part of message"
    )


def test_emoji_only_message_is_normal_text():
    text = (
        "[6/13/26, 7:20:53 PM] You: 🙏🏻\n"
        "[6/5/26, 9:33:55 PM] You: 😭😭😂"
    )
    result = WhatsAppParser().parse_text(text)

    assert [m.message_type for m in result.messages] == ["text", "text"]
    assert [m.is_media for m in result.messages] == [False, False]


def test_source_file_is_preserved():
    result = WhatsAppParser().parse_text(
        "[9/5/26, 7:31:22 PM] You: hello",
        source_file="friend_01.txt",
    )
    assert result.messages[0].source_file == "friend_01.txt"


def test_multiple_files_directory(tmp_path):
    (tmp_path / "chat_b.txt").write_text(
        "[9/5/26, 7:31:22 PM] B: hello", encoding="utf-8"
    )
    (tmp_path / "chat_a.txt").write_text(
        "[9/5/26, 7:31:22 PM] A: hello", encoding="utf-8"
    )
    (tmp_path / "ignore.md").write_text("not a chat", encoding="utf-8")

    results = WhatsAppParser().parse_directory(tmp_path)

    assert [r.source_file for r in results] == ["chat_a.txt", "chat_b.txt"]
    assert [r.messages[0].sender for r in results] == ["A", "B"]


def test_line_provenance_with_multiline_and_blank_line():
    text = (
        "[9/5/26, 7:31:22 PM] You: line 1\n"
        "line 2\n"
        "\n"
        "line 4\n"
        "[9/5/26, 7:32:22 PM] You: next"
    )
    result = WhatsAppParser().parse_text(text)

    assert result.messages[0].raw_line_start == 1
    assert result.messages[0].raw_line_end == 4
