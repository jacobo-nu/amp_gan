"""Agentes con modelo intercambiable (p. ej. un modelo local servido por Ollama).

Este paquete es independiente de ``amp_gan``: no importa Keras, torch ni TensorFlow y usa solo la
biblioteca estándar. La definición de cada agente vive en ``context/agents/<nombre>.md`` y es la
misma para cualquier modelo; lo único que cambia entre proveedores es el adaptador de
:mod:`amp_agents.providers`.

Módulos:

- :mod:`amp_agents.providers`: adaptadores de modelo (API compatible con OpenAI, proveedor guionizado).
- :mod:`amp_agents.tools`: herramientas acotadas y política de permisos.
- :mod:`amp_agents.loop`: bucle modelo -> herramientas -> modelo, con registro JSONL.
- :mod:`amp_agents.agents`: configuración (``agents.toml``) y ensamblado del prompt de sistema.
- :mod:`amp_agents.run`: línea de órdenes.
- :mod:`amp_agents.smoke_live`: comprobación en vivo contra un servidor local (no es un test de pytest).
"""
