# Proyecto_Final_Fuentes_Datos
Instrucciones para usuarios MAC:
1.- Clonar este repo en computadora
2.- Instalar XQuartz
3.- Abrir XQuartz-->Ajustes-->Seguridad-->Marca: “Allow connections from network clients"
4.- Reiniciar XQuartz
5.- En terminal de MAC ejecutar: export DISPLAY=:0
6.- Ejecutar también: xhost + 127.0.0.1
7.- Conectarse a repo desde terminal de computadora: Proyecto_Final_Fuentes_datos
8.- Construir imagen: docker build -t nombre_imagen -f resources/Dockerfile .
9.- Ejecutar imagen: docker run -e DISPLAY=host.docker.internal:0 -it nombre_imagen
