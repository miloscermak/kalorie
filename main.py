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
        
        prompt = """Jsi zkušený odborník na analýzu jídla a výživu. Tvým úkolem je analyzovat fotografii jídla a poskytnout detailní informace o něm. Zde je fotografie jídla, kterou budeš analyzovat:

<fotografie_jidla>
{{IMAGE}}
</fotografie_jidla>

Pečlivě si prohlédni všechny detaily zobrazené na fotografii. Zaměř se na ingredience, způsob přípravy, velikost porce a celkový vzhled jídla.

Před poskytnutím konečné odpovědi proveď důkladnou analýzu v následujících krocích. 

1. Popis jídla:
   - Popiš, co vidíš na fotografii, včetně textury, barvy a prezentace
   - Identifikuj hlavní ingredience
   - Odhadni způsob přípravy
   - Odhadni přesnou velikost porce

2. Kulturní kontext:
   - Zvaž možný původ jídla a jeho kulturní význam
   - Navrhni potenciální variace tohoto jídla

3. Návrh názvu:
   - Na základě pozorování navrhni vhodný český název pro toto jídlo
   - Ujisti se, že název je výstižný a popisný

4. Odhad kalorické hodnoty:
   - Zvaž viditelné ingredience a jejich přibližné množství
   - Vezmi v úvahu odhadnutou velikost porce
   - Odhadni přibližnou kalorickou hodnotu a zdůvodni svůj odhad

5. Základní informace:
   - Shrň klíčové informace o jídle (např. původ, typické použití, variace)

6. Zdravotní benefity:
   - Identifikuj potenciální pozitivní účinky jídla na zdraví
   - Zvaž nutriční hodnotu jednotlivých ingrediencí

7. Zdravotní rizika:
   - Zvaž možná zdravotní rizika spojená s konzumací tohoto jídla
   - Vezmi v úvahu alergeny, vysoký obsah tuku nebo cukru, apod.

Na základě své analýzy nyní poskytni strukturovanou odpověď v následujícím formátu:

Název jídla:
[Navržený název jídla v češtině]

Kalorická hodnota:
[Odhadovaná kalorická hodnota jídla v češtině, včetně zdůvodnění odhadu]

Poznámky:
[Základní informace o jídle]

Zdravotní benefity:
[Seznam potenciálních pozitivních účinků na zdraví]

Zdravotní rizika:
[Seznam možných zdravotních rizik]

Ujisti se, že tvá odpověď je v češtině a obsahuje všechny požadované sekce. Buď konkrétní a výstižný ve svých popisech a odhadech.
"""

        # Vytvoření zprávy pomocí nového API
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
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
