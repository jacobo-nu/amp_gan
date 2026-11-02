# Gobernanza

## Roles
- **Usuario**: fija objetivos, revisa y aprueba. Decide sobre cualquier acción externa.
- **Orquestador**: planifica, delega, valida y mantiene el roadmap.
- **Ejecutor**: implementa tareas acotadas dentro de `src/`.

## Reglas
1. El ejecutor solo recibe tareas acotadas con criterios de aceptación explícitos.
2. Solo el ejecutor escribe en `src/`; solo el orquestador modifica `context/`, con una excepción: el ejecutor puede escribir en `context/data/processed/` los datos que generen sus scripts. `context/data/raw/` y `context/references/` no los modifica el ejecutor.
3. Ningún cambio se da por válido sin haber sido ejecutado (notebooks/tests).
4. Todo experimento fija semillas y registra versiones de librerías.
5. Los datos (`context/data/`) se descargan por script; no se suben al repositorio. Tampoco se suben los PDF de `context/references/`: nada de `context/data/` ni `context/references/` se sincroniza, salvo sus `README.md` (índice y citas) y los `.gitkeep`.
6. Se respetan las licencias de las bases de datos y se citan las fuentes.

## Git
- `main` contiene siempre algo que funciona (tests y notebooks se ejecutan).
- Cambios de documentación (`context/`, README, TODO): commit directo en `main`.
- Código y experimentos (`src/`, variantes de GAN, notebooks): en ramas `feat/...` y `fix/...`, que se
  unen a `main` solo cuando el orquestador las ha validado y el usuario las aprueba.
- Los commits que genere el orquestador no se suben solos: el push lo hace el usuario.
- Commits pequeños y descriptivos (convención `tipo: mensaje`).
- **Push, PR y cualquier acción en GitHub requieren aprobación explícita del usuario.**

## Notebooks
- Se guardan con las salidas limpias salvo que el usuario pida lo contrario.
- La lógica reutilizable vive en `src/amp_gan/`; los notebooks la importan.

## Alcance ético
El proyecto se limita a modelar péptidos antimicrobianos con fines educativos y
de investigación abierta. No se generan ni optimizan secuencias para dañar a personas.
