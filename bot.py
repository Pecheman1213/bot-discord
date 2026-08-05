import discord
from discord.ext import commands
#from modelo import get_class
import os , random
from ultralytics import YOLO
from PIL import Image

# Crear el bot con prefijo de comando
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!p ', intents=intents)
modelo = YOLO("yolov8n.pt")

# Evento: El bot está listo
@bot.event
async def on_ready():
    print(f'Bot iniciado como {bot.user}')

# Comando: saludo
@bot.command(name='hola')
async def hola(ctx):
    await ctx.send(f'¡Hola {ctx.author.name}!')


# Funcion 'analizar' ; entrada: imagen ; salida : texto con los resultados.
@bot.command(name = 'analizar')
async def analizar(ctx):

    nombre_random = ''

    # Si el mensaje tiene archivos adjuntos.
    if ctx.message.attachments:

        # Capturar la primera imagen enviada.
        imagen_adjunta = ctx.message.attachments[0]

        #Listar nombre de imagenes existenes
        lista_archivos = os.listdir("imagenes")
        
        while True:
            # Crear un nombre random para la imagen
            nombre_random = "imagen" + str(random.randint(0,10000)) + ".png"

            # Se asegura de que no exista una imagen con el mismo nombre.
            if nombre_random not in lista_archivos:
                break

        # Guardar la imagen que el usuario adjunto.
        await imagen_adjunta.save(f"imagenes/{nombre_random}")
        
        # Abir imagen con pillow.
        cargar_imagen = Image.open("imagenes/"+ nombre_random)

        # Inferencia.
        resultados = modelo(cargar_imagen)

        # Objetos Detectados.
        lista_objetos = []
        count = 1

        for data in resultados:
            for box in data.boxes:
                porcentaje = float (box.conf[0])

                if porcentaje > 0.80:
                    cls = int(box.cls[0])
                    nombre = f"{str(count)+ '. ' + data.names[cls]}"
                    lista_objetos.append(nombre)
                    count += 1


        if len(lista_objetos) < 0:
            await ctx.send("No se detecto ningun objeto con un porcentaje de confianza mayor al 80%")

        else:
            await ctx.send("Esto fue lo que detecte:\n" + "\n".join(lista_objetos))


@bot.command(name="detectar")
async def detectar(ctx):

    if not ctx.message.attachments:
        await ctx.send("📸 Envíame una imagen para analizarla.")
        return

    imagen_adjunta = ctx.message.attachments[0]

    # Guardar imagen
    archivo = "imagenes/imagen.png"
    await imagen_adjunta.save(archivo)

    # Analizar imagen
    resultados = modelo(archivo, conf=0.50)

    # Dibujar los recuadros
    imagen_dibujada = resultados[0].plot()

    # Guardar imagen con las detecciones
    imagen_final = Image.fromarray(imagen_dibujada[:, :, ::-1])
    imagen_final.save("imagenes/resultado.png")

    # Contar objetos encontrados
    cantidad = len(resultados[0].boxes)

    if cantidad > 0:
        await ctx.send(
            "🤖 **¡Encontré " + str(cantidad) + " objetos!**",
            file=discord.File("imagenes/resultado.png")
        )
    else:
        await ctx.send("🔎 No encontré ningún objeto.")


@bot.command(name="ranking")
async def ranking(ctx):

    if not ctx.message.attachments:
        await ctx.send("📸 Envíame una imagen para analizarla.")
        return

    imagen_adjunta = ctx.message.attachments[0]

    # Guardar imagen
    archivo = "imagenes/ranking.png"
    await imagen_adjunta.save(archivo)

    # Analizar imagen
    resultados = modelo(archivo, conf=0.50)

    # Diccionario para contar objetos
    objetos = {}

    for box in resultados[0].boxes:

        cls = int(box.cls[0])
        nombre = resultados[0].names[cls]

        if nombre in objetos:
            objetos[nombre] += 1
        else:
            objetos[nombre] = 1

    # Ordenar de mayor a menor
    ranking = sorted(
        objetos.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if len(ranking) == 0:
        await ctx.send("🔎 No encontré ningún objeto.")
        return

    # Crear mensaje
    mensaje = "🏆 **RANKING DE OBJETOS**\n\n"

    posicion = 1

    for nombre, cantidad in ranking:
        mensaje += str(posicion) + ". **" + nombre + "** → " + str(cantidad) + "\n"
        posicion += 1

    await ctx.send(mensaje)



bot.run("TOKEN")
