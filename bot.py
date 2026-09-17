import discord
from discord.ext import commands
import os
import uuid
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

# Cargar token desde variable de entorno (más seguro que hardcodearlo)
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Crear el bot con prefijo de comando
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!p ', intents=intents)
modelo = YOLO("yolov8n.pt")

os.makedirs("imagenes", exist_ok=True)


async def guardar_imagen_adjunta(ctx, prefijo="imagen"):
    """Guarda el primer adjunto del mensaje y devuelve la ruta, o None si no hay adjuntos."""
    if not ctx.message.attachments:
        await ctx.send("📸 Envíame una imagen para analizarla.")
        return None

    adjunto = ctx.message.attachments[0]
    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}.png"
    ruta = os.path.join("imagenes", nombre)
    await adjunto.save(ruta)
    return ruta


@bot.event
async def on_ready():
    print(f'Bot iniciado como {bot.user}')


@bot.command(name='hola')
async def hola(ctx):
    await ctx.send(f'¡Hola {ctx.author.name}!')


@bot.command(name='analizar')
async def analizar(ctx):
    ruta = await guardar_imagen_adjunta(ctx, "analizar")
    if ruta is None:
        return

    try:
        cargar_imagen = Image.open(ruta)
        resultados = modelo(cargar_imagen)
    except Exception as e:
        await ctx.send(f"⚠️ Error al analizar la imagen: {e}")
        return

    lista_objetos = []
    for data in resultados:
        for box in data.boxes:
            porcentaje = float(box.conf[0])
            if porcentaje > 0.80:
                cls = int(box.cls[0])
                lista_objetos.append(data.names[cls])

    if not lista_objetos:
        await ctx.send("No detecté ningún objeto con una confianza mayor al 80%.")
    else:
        texto = "\n".join(f"{i+1}. {obj}" for i, obj in enumerate(lista_objetos))
        await ctx.send("Esto fue lo que detecté:\n" + texto)


@bot.command(name="detectar")
async def detectar(ctx):
    ruta = await guardar_imagen_adjunta(ctx, "detectar")
    if ruta is None:
        return

    try:
        resultados = modelo(ruta, conf=0.50)
    except Exception as e:
        await ctx.send(f"⚠️ Error al analizar la imagen: {e}")
        return

    imagen_dibujada = resultados[0].plot()
    ruta_resultado = ruta.replace(".png", "_resultado.png")
    Image.fromarray(imagen_dibujada[:, :, ::-1]).save(ruta_resultado)

    cantidad = len(resultados[0].boxes)

    if cantidad > 0:
        await ctx.send(
            f"🤖 **¡Encontré {cantidad} objetos!**",
            file=discord.File(ruta_resultado)
        )
    else:
        await ctx.send("🔎 No encontré ningún objeto.")


@bot.command(name="ranking")
async def ranking(ctx):
    ruta = await guardar_imagen_adjunta(ctx, "ranking")
    if ruta is None:
        return

    try:
        resultados = modelo(ruta, conf=0.50)
    except Exception as e:
        await ctx.send(f"⚠️ Error al analizar la imagen: {e}")
        return

    objetos = {}
    for box in resultados[0].boxes:
        cls = int(box.cls[0])
        nombre = resultados[0].names[cls]
        objetos[nombre] = objetos.get(nombre, 0) + 1

    ranking_ordenado = sorted(objetos.items(), key=lambda x: x[1], reverse=True)

    if not ranking_ordenado:
        await ctx.send("🔎 No encontré ningún objeto.")
        return

    mensaje = "🏆 **RANKING DE OBJETOS**\n\n"
    for posicion, (nombre, cantidad) in enumerate(ranking_ordenado, start=1):
        mensaje += f"{posicion}. **{nombre}** → {cantidad}\n"

    await ctx.send(mensaje)


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("No se encontró DISCORD_TOKEN. Definilo en un archivo .env")
    bot.run(TOKEN)
