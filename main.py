# =============================================================================
# LABORATÓRIO: Clone do ChatGPT com FastAPI + Modelos HuggingFace (LoRA)
# =============================================================================
# Estrutura de pastas esperada:
#
#   LLM_NR12/
#   ├── main.py
#   ├── dataset.jsonl
#   ├── static/
#   │   └── index.html
#   ├── lora/
#   │   ├── mt5/            ← adaptadores LoRA do google/mt5-small
#   │   ├── ptt5/           ← adaptadores LoRA do unicamp-dl/ptt5-v2-base
#   │   ├── tinyllama/      ← adaptadores LoRA do TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   │   └── tucano/         ← adaptadores LoRA do TucanoBR/Tucano-160m
#   └── tokenizer/
#       ├── mt5/            ← tokenizador salvo do mt5
#       ├── ptt5/           ← tokenizador salvo do ptt5
#       ├── tinyllama/      ← tokenizador salvo do tinyllama
#       └── tucano/         ← tokenizador salvo do tucano
#
# Se a subpasta do tokenizador não existir, o carregador faz fallback
# direto para o HuggingFace Hub.
# Se a subpasta de lora não existir, o modelo é ignorado sem erro fatal.
# =============================================================================

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from peft import PeftModel
import torch

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# APLICAÇÃO FASTAPI
# =============================================================================
app = FastAPI(
    title="LLM Lab – 4 Modelos LoRA",
    description="API para interagir com os 4 modelos fine-tunados com LoRA",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CATÁLOGO DE MODELOS
# =============================================================================
# Caminhos relativos à pasta do projeto (LLM_NR12/).
#
# arquitetura:
#   "seq2seq" → AutoModelForSeq2SeqLM  (mt5, ptt5)
#   "causal"  → AutoModelForCausalLM   (tinyllama, tucano)
#
# prompt_fn:
#   Recebe (instruction, input_text) e devolve o prompt formatado.
#   Deve bater com o template usado durante o fine-tuning de cada modelo.

MODEL_CATALOG = {
    # ── google/mt5-small ──────────────────────────────────────────────────
    "mt5": {
        "id":          "mt5",
        "nome":        "mT5-small (Seq2Seq)",
        "descricao":   "Google mT5-small multilíngue, fine-tunado com LoRA.",
        "arquitetura": "seq2seq",
        "base_model":  "google/mt5-small",
        "lora_path":   "./lora/mt5_lora",
        "tok_path":    "./tokenizer/mt5_tokenizer",
        # Formato usado no notebook: "Instruction: {inst}" ou "Instruction: {inst}\nInput: {inp}"
        "prompt_fn": lambda inst, inp: (
            f"Instruction: {inst}\nInput: {inp}"
            if inp.strip()
            else f"Instruction: {inst}"
        ),
    },

    # ── unicamp-dl/ptt5-v2-base ───────────────────────────────────────────
    "ptt5": {
        "id":          "ptt5",
        "nome":        "PTT5-v2-base (Seq2Seq)",
        "descricao":   "Modelo encoder-decoder português da Unicamp, fine-tunado com LoRA.",
        "arquitetura": "seq2seq",
        "base_model":  "unicamp-dl/ptt5-v2-base",
        "lora_path":   "./lora/ptt5_lora",
        "tok_path":    "./tokenizer/ptt5_tokenizer",
        # ptt5 foi treinado com prefixo em português
        "prompt_fn": lambda inst, inp: (
            f"Instrução: {inst}\nEntrada: {inp}"
            if inp.strip()
            else f"Instrução: {inst}"
        ),
    },

    # ── TinyLlama/TinyLlama-1.1B-Chat-v1.0 ──────────────────────────────
    "tinyllama": {
        "id":          "tinyllama",
        "nome":        "TinyLlama-1.1B (Causal LM)",
        "descricao":   "Llama compacto multilíngue, fine-tunado com LoRA.",
        "arquitetura": "causal",
        "base_model":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_path":   "./lora/tinyllama_lora",
        "tok_path":    "./tokenizer/tinyllama_tokenizer",
        "prompt_fn": lambda inst, inp: (
            f"Instruction: {inst}\nInput: {inp}\nOutput:"
            if inp.strip()
            else f"Instruction: {inst}\nOutput:"
        ),
    },

    # ── TucanoBR/Tucano-160m ─────────────────────────────────────────────
    "tucano": {
        "id":          "tucano",
        "nome":        "Tucano-160m (Causal LM)",
        "descricao":   "GPT-2 em português da CEIA, fine-tunado com LoRA.",
        "arquitetura": "causal",
        "base_model":  "TucanoBR/Tucano-160m",
        "lora_path":   "./lora/tucano_lora",
        "tok_path":    "./tokenizer/tucano_tokenizer",
        "prompt_fn": lambda inst, inp: (
            f"Instruction: {inst}\nInput: {inp}\nOutput:"
            if inp.strip()
            else f"Instruction: {inst}\nOutput:"
        ),
    },
}

# =============================================================================
# DICIONÁRIO GLOBAL DE MODELOS CARREGADOS
# Chave → model_id  |  Valor → { model, tokenizer, arquitetura, prompt_fn }
# =============================================================================
MODELS: dict = {}

# =============================================================================
# CARREGADOR GENÉRICO
# =============================================================================

def carregar_modelo(model_id: str) -> dict | None:
    """
    Carrega um modelo a partir do catálogo MODEL_CATALOG.

    Fluxo:
      1. Lê as configurações do catálogo.
      2. Verifica se a pasta de adaptadores LoRA existe; pula se não existir.
      3. Carrega o tokenizador local (ou faz fallback para o HuggingFace Hub).
      4. Carrega o modelo base (Seq2Seq ou Causal).
      5. Aplica os adaptadores LoRA com PeftModel.from_pretrained().
      6. Coloca em modo eval() e retorna o dicionário.
    """
    cfg = MODEL_CATALOG.get(model_id)
    if cfg is None:
        logger.error(f"Model ID '{model_id}' não encontrado no catálogo.")
        return None

    lora_path = cfg["lora_path"]

    # ── Verifica se os adaptadores LoRA existem ───────────────────────────
    if not os.path.isdir(lora_path):
        logger.warning(
            f"  ⚠ [{model_id}] Diretório LoRA '{lora_path}' não encontrado. "
            "Pulando este modelo."
        )
        return None

    logger.info(f"  → Carregando [{model_id}] ({cfg['arquitetura']}) ...")

    # ── Tokenizador ───────────────────────────────────────────────────────
    tok_path = cfg["tok_path"]
    tok_source = tok_path if os.path.isdir(tok_path) else cfg["base_model"]
    if tok_source == cfg["base_model"]:
        logger.info(
            f"    Tokenizador local não encontrado em '{tok_path}'. "
            f"Baixando do Hub: {cfg['base_model']}"
        )

    tokenizer = AutoTokenizer.from_pretrained(tok_source)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Modelo base ───────────────────────────────────────────────────────
    # Verifica se existe um modelo salvo localmente na pasta Modelos/
    # Tenta primeiro o nome com sufixo usado nos notebooks, depois sem sufixo
    # (opcional – economiza download ao reiniciar o servidor)
    local_model_path = (
        f"./Modelos/{model_id}_model"
        if os.path.isdir(f"./Modelos/{model_id}_model")
        else f"./Modelos/{model_id}"
    )
    model_source = (
        local_model_path
        if os.path.isdir(local_model_path)
        else cfg["base_model"]
    )
    if model_source == cfg["base_model"]:
        logger.info(f"    Modelo base: Hub ({cfg['base_model']})")
    else:
        logger.info(f"    Modelo base: local ({local_model_path})")

    if cfg["arquitetura"] == "seq2seq":
        base = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,   # CPU: float32 evita erros de precisão
        )
    else:  # causal
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )

    # ── Aplica adaptadores LoRA ───────────────────────────────────────────
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    logger.info(f"  ✓ [{model_id}] pronto!")
    return {
        "model":       model,
        "tokenizer":   tokenizer,
        "arquitetura": cfg["arquitetura"],
        "prompt_fn":   cfg["prompt_fn"],
    }


# =============================================================================
# STARTUP: carrega todos os modelos disponíveis
# =============================================================================

@app.on_event("startup")
async def startup_event():
    global MODELS
    logger.info("=" * 60)
    logger.info("  INICIANDO SERVIDOR – carregando modelos LoRA...")
    logger.info("=" * 60)

    for model_id in MODEL_CATALOG:
        resultado = carregar_modelo(model_id)
        if resultado is not None:
            MODELS[model_id] = resultado

    logger.info("=" * 60)
    logger.info(
        f"  ✓ {len(MODELS)} modelo(s) disponível(is): {list(MODELS.keys())}"
    )
    logger.info("=" * 60)


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ChatRequest(BaseModel):
    modelo:      str
    mensagem:    str
    max_tokens:  Optional[int]   = 150
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    resposta:       str
    modelo:         str
    tokens_gerados: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    """
    GET /modelos
    Retorna apenas os modelos que foram carregados com sucesso.
    O front-end usa este endpoint para popular o dropdown.
    """
    disponiveis = [
        {
            "id":        cfg["id"],
            "nome":      cfg["nome"],
            "descricao": cfg["descricao"],
        }
        for key, cfg in MODEL_CATALOG.items()
        if key in MODELS
    ]
    return {"modelos": disponiveis}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat
    Gera uma resposta usando o modelo selecionado.
    Suporta Seq2Seq (mt5, ptt5) e Causal LM (tinyllama, tucano).
    """
    # ── Validações ────────────────────────────────────────────────────────
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{request.modelo}' não disponível. "
                f"Disponíveis: {list(MODELS.keys())}"
            ),
        )

    if not request.mensagem.strip():
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    entry        = MODELS[request.modelo]
    model        = entry["model"]
    tokenizer    = entry["tokenizer"]
    arquitetura  = entry["arquitetura"]
    prompt_fn    = entry["prompt_fn"]

    # Monta o prompt no formato que o modelo foi treinado
    prompt = prompt_fn(request.mensagem, "")

    logger.info(f"[CHAT] modelo={request.modelo} | prompt={prompt[:80]}...")

    try:
        if arquitetura == "seq2seq":
            resposta = _gerar_seq2seq(
                model, tokenizer, prompt,
                request.max_tokens, request.temperatura,
            )
        else:
            resposta = _gerar_causal(
                model, tokenizer, prompt,
                request.max_tokens, request.temperatura,
            )

        if not resposta:
            resposta = "[O modelo não gerou texto. Tente aumentar max_tokens.]"

        tokens_gerados = len(tokenizer.encode(resposta))
        logger.info(f"  ✓ {tokens_gerados} tokens gerados")

        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tokens_gerados=tokens_gerados,
        )

    except Exception as e:
        logger.error(f"Erro na geração [{request.modelo}]: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {e}")


# =============================================================================
# FUNÇÕES DE GERAÇÃO
# =============================================================================

def _gerar_seq2seq(
    model, tokenizer, prompt: str,
    max_new_tokens: int, temperature: float,
) -> str:
    """
    Geração para modelos encoder-decoder (mt5, ptt5).
    O encoder lê o prompt; o decoder gera a resposta.
    A saída decodificada NÃO inclui o prompt — apenas a resposta.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192,
        padding=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # evita temperatura 0
            do_sample=temperature > 0.1,
            top_p=0.9,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _gerar_causal(
    model, tokenizer, prompt: str,
    max_new_tokens: int, temperature: float,
) -> str:
    """
    Geração para modelos Causal LM (tinyllama, tucano).
    A saída inclui o prompt inteiro — extraímos apenas a parte após "Output:".
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove o prompt — pega apenas o que vem depois de "Output:"
    if "Output:" in full_text:
        return full_text.split("Output:")[-1].strip()

    # Fallback: remove o prompt manualmente pelo comprimento
    return full_text[len(prompt):].strip()


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade": len(MODELS),
    }


# =============================================================================
# FRONT-END ESTÁTICO
# =============================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
