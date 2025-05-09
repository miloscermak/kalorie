from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import anthropic
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("VAROVÁNÍ: API klíč nebyl nalezen v .env souboru")
elif not api_key.startswith("sk-ant-"):
    print("VAROVÁNÍ: API klíč nemá správný formát")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Načtení a zakódování obrázku do base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Kontrola velikosti souboru
        if len(contents) > 20 * 1024 * 1024:  # 20MB limit
            return {"error": "Soubor je příliš velký. Maximální velikost je 20MB."}
            
        # Kontrola typu souboru
        if file.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "Podporované formáty jsou pouze JPEG a PNG."}
        
        prompt = """Prohlédni si pozorně následující fotografii jídla:

        <image>
        {{IMAGE}}
        </image>

        Pečlivě si prohlédni všechny detaily zobrazené na fotografii. Zaměř se na ingredience, způsob přípravy, velikost porce a celkový vzhled jídla.

        Na základě svého pozorování proveď následující úkoly:

        1. Navrhni vhodný název pro toto jídlo v češtině. Název by měl být výstižný a popisný.

        2. Odhadni přibližnou kalorickou hodnotu zobrazeného jídla. Vezmi v úvahu viditelné ingredience, velikost porce a předpokládaný způsob přípravy.

        Svou odpověď napiš v následujícím formátu:

        <odpoved>
        <nazev_jidla>
        [Zde uveď navržený název jídla v češtině]
        </nazev_jidla>

        <kaloricka_hodnota>
        [Zde uveď odhadovanou kalorickou hodnotu jídla v češtině, včetně zdůvodnění svého odhadu]
        </kaloricka_hodnota>
        </odpoved>"""

        # Vytvoření zprávy pomocí nového API
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system="Jsi expert na analýzu jídla a kalorických hodnot.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.replace("{{IMAGE}}", "")
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": file.content_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )

        return {"response": response.content[0].text}
        
    except Exception as e:
        print(f"Chyba při zpracování: {str(e)}")  # Pro debugging
        return {"error": f"Došlo k chybě při analýze: {str(e)}"} 