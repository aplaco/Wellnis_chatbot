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
one_shot_prompt = """
### Instruction:
너무 무기력해.

### Response:
증상: 무기력증 무기력감은 삶의 목적과 의미를 잃고, 아무 일도 하지 못하는 상태를 말합니다. 이 증상은 우울증과 함께 나타날 수 있지만, 우울감만으로는 설명이 안되는 경우가 많습니다. 무기력감은 수면 문제, 식욕 변화, 집중력 저하 등의 증상이 동반될 수 있습니다.

### Instruction:
{}

### Response:
{}"""


@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    # 사용자 입력 저장
    chat_history.append({"user": user_input, "bot": None})
    
    # 프롬프트에 사용자 입력 삽입
    prompt = one_shot_prompt.format(user_input, "")

    # 추론을 위한 입력 준비
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")  # 텐서를 PyTorch 형식으로 변환하고 GPU로 이동

    # TextStreamer 초기화
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)

    # 모델을 사용하여 텍스트 생성 및 스트리밍 출력
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        do_sample=True,  # 샘플링을 활성화하여 다양성 부여 (옵션)
        temperature=0.7,  # 생성의 창의성 조절 (옵션)
        top_p=0.9  # 핵심 확률 질량을 설정하여 토큰 선택 (옵션)
    )

    # 답변만 추출
    generated_text = tokenizer.decode(_[0], skip_special_tokens=True)
    answer = generated_text.split("### Response:")[-1].strip()

    # 마지막 대화에 응답 추가
    chat_history[-1]["bot"] = answer

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)