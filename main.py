from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from unsloth import FastLanguageModel
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 전체 세션 동안 유지되는 대화 기록
chat_history = []

# 저장된 경로 지정 및 모델 로딩 (서버 시작 시 1회만 실행)
save_directory = "/home/alpaco/chat_bot/wellnis"

# 모델과 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_directory,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,  # 양자화 옵션을 동일하게 설정
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# 모델에게 주어질 텍스트의 형식을 정의
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    # 사용자 입력 저장
    chat_history.append({"user": user_input, "bot": None})

    # 추론을 위한 입력 준비
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "너무 우울해.", # 인스트럭션 (명령어)
            "", # 출력 - 생성할 답변을 비워둠
        )
    ], return_tensors = "pt").to("cuda")  # 텐서를 PyTorch 형식으로 변환하고 GPU로 이동

    from transformers import TextStreamer  # 텍스트 스트리밍을 위한 TextStreamer 임포트

    text_streamer = TextStreamer(tokenizer)  # 토크나이저를 사용하여 스트리머 초기화

    # 모델을 사용하여 텍스트 생성 및 스트리밍 출력
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)  # 최대 128개의 새로운 토큰 생성

    # 답변만 추출
    generated_text = tokenizer.decode(_[0], skip_special_tokens=True)
    answer = generated_text.split("### Response:")[1].strip()

    # 마지막 대화에 응답 추가
    chat_history[-1]["bot"] = answer

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)