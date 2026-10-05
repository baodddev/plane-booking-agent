# LangChain Plane Booking Agent

Demo một **agent đặt vé máy bay** bằng LangChain, minh hoạ ý tưởng:

> **Agent = Model + Harness**
>
> - **Model** (LLM) quyết định gọi tool nào tiếp theo.
> - **Harness** (code Python thường) quyết định lời gọi đó có được phép không, và việc đã *thật sự* xong chưa.

Toàn bộ tool và model đều là **mock** (dữ liệu giả, model chạy theo kịch bản), nên **không cần API key và không cần mạng**.

Yêu cầu của người dùng trong mọi demo:

```
Book one ticket SGN -> DAD on 2026-10-07, departing before 12:00, price at most 2,000,000 VND.
```

## Cấu trúc project

| File | Nội dung |
|------|----------|
| `flight_agent.py` | Agent kiểu **ReAct** + harness cơ bản (4 ý tưởng: constraints là data, kiểm tra quyền, kiểm tra "done" bằng code, handoff cho người). |
| `flight_agent_plan_then_execute.py` | Kiểu **Plan-then-Execute**: model viết cả kế hoạch trong 1 lần gọi → người duyệt → code chạy từng bước. |
| `flight_agent_hybrid.py` | Kiểu **Hybrid**: lập kế hoạch trước, nếu một bước bị chặn thì chuyển sang ReAct để tự phục hồi. |
| `flight_agent_failure_mode.py` | 4 **lỗi kinh điển** của agent (vòng lặp vô hạn, bịa dữ liệu, quên mục tiêu, hỏng trạng thái) và cách harness bắt từng lỗi. Có thể bật/tắt harness để so sánh. |
| `FLOW.md` | Sơ đồ luồng và giải thích từng khối code. |
| `requirements.txt` | Các thư viện cần cài. |

## Cài đặt

Cần **Python ≥ 3.10** (khuyên dùng bản CPython chính thức từ python.org).

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Lưu ý Windows:** nếu `python` trên máy bạn là bản của MSYS2/MinGW (`C:\msys64\...`),
> `pip install` có thể lỗi khi build `uuid-utils` (không có wheel sẵn cho MinGW).
> Hãy tạo venv bằng CPython chính thức, ví dụ:
> `C:\Users\<you>\AppData\Local\Python\bin\python3.14.exe -m venv .venv`

## Chạy

Mỗi file là một chương trình tương tác — chạy rồi chọn kịch bản bằng bàn phím.

```bash
python flight_agent.py                   # 1 = success, 2 = fail
python flight_agent_plan_then_execute.py # 1 = success, 2 = fail; sau đó duyệt plan (y/n)
python flight_agent_hybrid.py            # 1 = success, 2 = recovery, 3 = fail; sau đó duyệt plan (y/n)
python flight_agent_failure_mode.py      # 1-4 = loại lỗi; sau đó bật harness? (y/n)
```

Chạy không tương tác (để test nhanh), dùng pipe:

```bash
printf '2\ny\n' | python flight_agent_hybrid.py
```

## Kết quả mong đợi (đã chạy thử)

| Lệnh | Lựa chọn | Kết quả |
|------|----------|---------|
| `flight_agent.py` | 1 success | `DONE` – đặt và trả tiền VN122 |
| `flight_agent.py` | 2 fail | `FAILED` – harness chặn VJ604 (quá giá) → handoff |
| `flight_agent_plan_then_execute.py` | 1, y | `DONE` |
| `flight_agent_plan_then_execute.py` | 2, n | `FAILED` – plan #2 rỗng → handoff |
| `flight_agent_plan_then_execute.py` | 2, y | `FAILED` – bước `book_seat VJ604` bị chặn → dừng |
| `flight_agent_hybrid.py` | 1, y | `DONE` |
| `flight_agent_hybrid.py` | 2, y | `DONE` – plan chọn QH118 bị chặn, ReAct phục hồi bằng VN122 |
| `flight_agent_hybrid.py` | 3, y | `FAILED` – không chuyến nào hợp lệ → handoff |
| `flight_agent_hybrid.py` | bất kỳ, n | `FAILED` – `plan_rejected` |
| `flight_agent_failure_mode.py` | 1–4, n | Không có harness: tin lời model (lặp 10 lần / báo vé giả / đặt sai chuyến / tưởng không có chuyến bay) |
| `flight_agent_failure_mode.py` | 1–4, y | Có harness: `FAILED` + `stop_reason` chỉ đúng lỗi + handoff |

Ví dụ output của `flight_agent_hybrid.py` kịch bản recovery:

```
--- plan (recovery) ---
  Step 1: book_seat({'flight': 'QH118'})
  Step 2: pay({'code': '$booking_code'})
  Step 3: get_booking({'code': '$booking_code'})
Human reviewer - approve this plan? (y/n): y
  Plan step 1 blocked: QH118 does not meet: Book one ticket SGN -> DAD ...

--- ReAct recovery ---
ReAct response: Recovered by selecting valid flight VN122.

--- trace ---
[1] search_flights(...) -> ok
[2] book_seat({'flight': 'QH118'}) -> denied
[3] search_flights(...) -> ok
[4] book_seat({'flight': 'VN122'}) -> ok
[5] pay({'code': 'VN122-12A'}) -> ok
[6] get_booking({'code': 'VN122-12A'}) -> ok

--- result ---
{ "result": "DONE", "booking": [ { "code": "VN122-12A", "paid": true, ... } ] }
```

## Dùng LLM thật thay cho model giả

Các hàm `fake_model()` / `scripted_models()` chỉ để demo chạy không cần key.
Muốn dùng LLM thật, cài thêm provider (ví dụ `pip install langchain-anthropic`) rồi thay model:

```python
from langchain.chat_models import init_chat_model
model = init_chat_model("anthropic:claude-sonnet-5-5")   # cần ANTHROPIC_API_KEY
```

Harness giữ nguyên — đó chính là ý tưởng: dù model nào, code vẫn là người gác cổng.

Xem `FLOW.md` để hiểu chi tiết luồng chạy và từng khối code.
