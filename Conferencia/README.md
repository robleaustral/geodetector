# GEODETECTOR_JCC2026

Versión preliminar del artículo en español para JCC 2026 / WSPRP, usando formato IEEE Conference.

## Archivos

- `paper.tex`: manuscrito principal en LaTeX.
- `sample.bib`: archivo BibTeX con las referencias.
- `IEEEtran.cls`: clase IEEE requerida por el template.
- `figuras/`: contiene archivos PDF reservados para las figuras que deben ser reemplazados por las figuras definitivas de la tesis.
- `compile.sh`: script de compilación.

## Figuras reservadas

- `figuras/figura2.pdf`: Pipeline general.
- `figuras/figura5.pdf`: Ejemplos de detección.
- `figuras/figura7.pdf`: Ejemplo end-to-end.
- `figuras/figura8.pdf`: Mapa de detecciones.
- `figuras/figura9.pdf`: Limitaciones / falsos positivos.

Actualmente las figuras son marcadores de posición. Para la versión final, reemplazar cada PDF por la figura correspondiente exportada desde la tesis, conservando el mismo nombre.

## Compilación

```bash
chmod +x compile.sh
./compile.sh
```

El script ejecuta `pdflatex`, `bibtex` y dos pasadas adicionales de `pdflatex`.

## Nota editorial

El artículo fue reescrito como paper IEEE en español, no como una reducción literal de la tesis. Conserva los elementos solicitados: Figura 2, Tabla 1, métricas resumidas de entrenamiento, Figura 5, Tabla 3, Tabla 5, resultado end-to-end de 57,1%, Figura 7, Figura 8 y Figura 9.
