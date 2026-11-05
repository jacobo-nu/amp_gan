"""Agentes con modelo intercambiable (p. ej. un modelo local servido por Ollama).

Este paquete es independiente de ``amp_gan``: no importa Keras, torch ni TensorFlow y usa solo la
biblioteca estándar. La definición de cada agente vive en ``context/agents/<nombre>.md`` y es la
misma para cualquier modelo; lo único que cambia entre proveedores es el adaptador de
:mod:`amp_agents.providers`.

Módulos:

- :mod:`amp_agents.providers`: adaptadores de modelo (API compatible con OpenAI, proveedor guionizado).
- :mod:`amp_agents.tools`: herramientas acotadas y política de permisos.
- :mod:`amp_agents.loop`: bucle modelo -> herramientas -> modelo (``run_agent``) y modo de una sola petición sin
  herramientas (``run_single_shot``), con registro JSONL.
- :mod:`amp_agents.agents`: configuración (``agents.toml``) y ensamblado del prompt de sistema.
- :mod:`amp_agents.retrieval`: recuperador de documentación (troceado, índice de embeddings, línea base léxica,
  herramienta ``search_docs``).
- :mod:`amp_agents.verifiers`: comprobación automática de la salida de los roles (JSON, citas, líneas, cifras).
- :mod:`amp_agents.run`: línea de órdenes (``--agent`` ejecutor, localizador, verificador documental, revisor de
  código, resumidor o recuperador).
- :mod:`amp_agents.eval_roles`: evaluación en vivo de los roles con protocolo y criterios fijos (no es un test).
- :mod:`amp_agents.smoke_live`: comprobación en vivo contra un servidor local (no es un test de pytest).
"""
