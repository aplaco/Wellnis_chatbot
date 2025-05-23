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
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=save_directory,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    # 사용자 입력 저장
    chat_history.append({"user": user_input, "bot": None})

    # 토큰화
    inputs = tokenizer(user_input, return_tensors="pt").to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 응답 디코딩 및 사용자 입력 이후 부분만 추출
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(user_input):].strip()

    # 마지막 대화에 응답 추가
    chat_history[-1]["bot"] = answer

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)