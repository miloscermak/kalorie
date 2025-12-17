from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Gemini client (lazy initialization)
_client = None

def get_gemini_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client

STUDIO_PORTRAIT_PROMPT = """Transform this person's photo into a professional studio portrait with the following specifications:

STYLE: Clean Studio Portrait
- Ultra-realistic studio portrait with soft lighting
- Natural skin texture preserved
- Minimal, clean aesthetic

SUBJECT:
- Portrait type: tight face centered framing
- Expression: neutral, calm
- Lighting: soft diffused studio lighting
- Skin texture: realistic, natural
- If visible, wardrobe should appear in solid neutral colors

BACKGROUND:
- Seamless backdrop
- Color: light grey or white (#EDEDED)
- Style: minimal and clean

TEXTURE & FINISH:
- Subtle grain
- Fine digital noise
- Minimal compression artifacts
- Matte editorial finish

COLOR PALETTE:
- Neutral soft tones
- Medium contrast
- No strong accent colors

CAMERA SIMULATION:
- 85mm portrait lens effect
- Shallow depth of field
- Focus on eyes
- Straight-on angle

OUTPUT:
- Professional studio portrait format
- Aspect ratio: 3:4
- High resolution quality

Keep the person's identity and likeness accurate while applying professional studio portrait aesthetics."""


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_portrait(file: UploadFile = File(...)):
    try:
        # Read and validate the uploaded image
        contents = await file.read()

        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"error": "Soubor je příliš velký. Maximální velikost je 10MB."}
            )

        # Check file type
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Podporované formáty jsou JPEG, PNG a WebP."}
            )

        # Open image with PIL to validate and get dimensions
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            # Re-open after verify
            image = Image.open(io.BytesIO(contents))
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "Nepodařilo se načíst obrázek. Zkontrolujte, že soubor není poškozený."}
            )

        # Prepare image for Gemini
        image_part = types.Part.from_bytes(
            data=contents,
            mime_type=file.content_type
        )

        # Generate portrait using Gemini
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[STUDIO_PORTRAIT_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        # Extract generated image from response
        generated_image_data = None
        response_text = None

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                generated_image_data = part.inline_data.data
            elif part.text is not None:
                response_text = part.text

        if generated_image_data is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Nepodařilo se vygenerovat portrét. Zkuste to prosím znovu."}
            )

        # Convert to base64 for frontend
        generated_base64 = base64.b64encode(generated_image_data).decode("utf-8")

        # Also send original image for comparison
        original_base64 = base64.b64encode(contents).decode("utf-8")

        return {
            "success": True,
            "original_image": f"data:{file.content_type};base64,{original_base64}",
            "generated_image": f"data:image/png;base64,{generated_base64}",
            "message": response_text or "Studiový portrét byl úspěšně vytvořen!"
        }

    except ValueError as e:
        if "GEMINI_API_KEY" in str(e):
            print(f"Chyba konfigurace: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": "Služba není správně nakonfigurována. Kontaktujte administrátora."}
            )
        raise
    except Exception as e:
        print(f"Chyba při generování: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Došlo k chybě při generování portrétu: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
