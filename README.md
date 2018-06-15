# Clustering de canciones mediante Fourier

*Este repositorio está enfocado a una charla que quiero dar en la PyCon ES 2018. 
El contenido del mismo aún está en un repositorio privado, pero se hará público
en cuanto esté terminado.*

## Resumen
Teniendo un conjunto de canciones, ¿cuáles son las más similares entre sí?
¿Podremos crear playlists de canciones que tengan un ritmo similar? Analizando
solo las frecuencias, identificaremos canciones similares y crearemos un
modelo de machine learning para encontrar nuevas canciones que os gusten.

### Ejemplos

Distintas técnicas y métricas se están probando actualmente.
En la carpeta
[imagenes](https://github.com/cperales/Fourier-classifying-songs/tree/master/imagenes)
se puede ver un ejemplo, donde automáticamente se han asignado canciones
de Miley Cyrus a un clúster 

![alt text](https://raw.githubusercontent.com/cperales/Fourier-classifying-songs/master/imagenes/cluster_1/Malibu_Miley_Cyrus.png "Malibu")

![alt text](https://raw.githubusercontent.com/cperales/Fourier-classifying-songs/master/imagenes/cluster_1/You're_gonna_make_me_lonesome_when_you_go_Miley_Cyrus.png "You're gonna make me lonesome when you go")

y canciones de The Cranberries (y una de los Beatles) a otro.

![alt text](https://raw.githubusercontent.com/cperales/Fourier-classifying-songs/master/imagenes/cluster_2/Zombie_The_Cranberries.png "Zombie - The Cranberries")
![alt text](https://raw.githubusercontent.com/cperales/Fourier-classifying-songs/master/imagenes/cluster_2/Dreams_The_Cranberries.png "Dreams - The Cranberries")
![alt text](https://raw.githubusercontent.com/cperales/Fourier-classifying-songs/master/imagenes/cluster_2/Here_Comes_the_Sun_The_Beatles.png "Here Comes the Sun - The Beatles")

Se puede ver, además, que las gráficas de las canciones de ambos clústers
son parecidas entre sí pero diferentes a las del otro grupo.

## Motivación
Quienes tengan una cuenta en **Spotify** que usen a menudo sabrán que cada
semana te recomiendan nuevas canciones, bajo la lista *Descubrimiento Semanal*.
Como **Spotify** categoriza las canciones en función de
[parámetros](https://www.theverge.com/tldr/2018/2/5/16974194/spotify-recommendation-algorithm-playlist-hack-nelson)
como la *energía*, *instrumentalidad*, etc, existe por detrás un proceso de clasificación en base a estos parámetros.

Pero, sin acceso a estos parámetros, ¿cómo podríamos encontrar canciones similares, y agruparlas? Como
[las notas musicales tienen frecuencias asociadas](https://www.intmath.com/trigonometric-graphs/music.php),
esta propuesta se fundamenta en pasar de series temporales a series frecuenciales, y agrupar estas series de
frecuencias usando varias técnicas y métricas de distancia.

## De tiempo a frecuencias
El paso de tiempo a frecuencia se puede conseguir mediante una función matemática llamada
[transformada de Fourier](https://es.wikipedia.org/wiki/Transformada_de_Fourier). La transformada
de Fourier es una [aplicación lineal](https://math.stackexchange.com/questions/140788/how-is-the-fourier-transform-linear)
que permite pasar del dominio de los tiempos al dominio de las frecuencias. Esto es, si tenemos un sonido registrado,
podemos encontrar la frecuencias a las que vibra a través de una transformada de Fourier, sin perder información. Esta
función está implementada en la librería **numpy**.
