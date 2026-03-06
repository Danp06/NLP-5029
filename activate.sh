#!/bin/bash

# Objetivo: Activar automáticamente un entorno virtual si existe, o crear uno nuevo.
# Al final, mostrar al usuario el resultado de "which pip" y "which python"
# para confirmar que NO están en /usr/bin, sino en el entorno virtual apropiado,
# y emitir una advertencia si siguen apuntando al sistema global.

virtualenvs=()
virtualenv_names=()

# Buscar todos los entornos virtuales en subcarpetas (incluyendo carpetas ocultas)
for d in .*/ */ ; do
    # Saltar las carpetas . y ..
    if [[ "$d" == "./" || "$d" == "../" ]]; then
        continue
    fi
    if [ -f "${d}bin/activate" ]; then
        virtualenvs+=("${d}")
        virtualenv_names+=("${d%/}")
    fi
done

# Revisar si hay un entorno virtual en la carpeta actual (bin/activate)
if [ -f "bin/activate" ]; then
    virtualenvs+=("")
    virtualenv_names+=("(carpeta actual)")
fi

activate_and_check() {
    # $1 es la ruta relativa al entorno virtual ("" = carpeta actual)
    if [ -z "$1" ]; then
        echo "Activando entorno virtual en la carpeta actual."
        source "bin/activate"
    else
        echo "Activando entorno virtual: $1"
        source "${1}bin/activate"
    fi

    echo ""
    echo "🧪 Verificación de entorno:"
    pip_path="$(which pip 2>/dev/null)"
    python_path="$(which python 2>/dev/null)"
    echo "which pip --> $pip_path"
    echo "which python --> $python_path"
    warn=false
    if [[ "$pip_path" == "/usr/bin/pip" || "$pip_path" == "/usr/local/bin/pip" ]]; then
        warn=true
    fi
    if [[ "$python_path" == "/usr/bin/python" || "$python_path" == "/usr/bin/python3" || "$python_path" == "/usr/local/bin/python" || "$python_path" == "/usr/local/bin/python3" || -z "$python_path" ]]; then
        warn=true
    fi
    if $warn; then
        echo "⚠️ ADVERTENCIA: 'pip' o 'python' corresponde al sistema global (/usr/bin)."
        echo "       El entorno virtual podría NO estar activado correctamente."
        echo "       (Recuerda ejecutar este script con 'source')."
    fi
}

if [ "${#virtualenvs[@]}" -eq 0 ]; then
    echo "No se encontró ningún entorno virtual en la carpeta actual ni en subcarpetas."
    read -p "¿Deseas crear un nuevo entorno virtual llamado '.venv' en la carpeta actual? (s/n): " create_env_choice
    if [[ "$create_env_choice" =~ ^[Ss]$ ]]; then
        echo "Creando entorno virtual '.venv'..."

        # Detectar binario de python a usar de acuerdo al sistema
        if command -v python3 &>/dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &>/dev/null; then
            PYTHON_CMD="python"
        else
            echo "No se encontró un binario de python en el PATH."
            echo "Por favor, instala Python (3.8 o superior recomendado) e intenta de nuevo."
            return 1 2>/dev/null || exit 1
        fi

        echo "Usando $PYTHON_CMD para crear el entorno virtual."

        "$PYTHON_CMD" -m venv .venv
        if [ $? -eq 0 ]; then
            echo "Entorno '.venv' creado exitosamente."
            echo "Activando entorno virtual: ./.venv"
            activate_and_check "./.venv/"
        else
            echo "Error al crear el entorno virtual '.venv'."
            return 1 2>/dev/null || exit 1
        fi
    else
        echo "No se creó ningún entorno virtual. Saliendo."
        return 1 2>/dev/null || exit 1
    fi
elif [ "${#virtualenvs[@]}" -eq 1 ]; then
    activate_and_check "${virtualenvs[0]}"
else
    echo "Se encontraron múltiples entornos virtuales:"
    for i in "${!virtualenvs[@]}"; do
        echo "$((i+1))) ${virtualenv_names[$i]}"
    done
    while true; do
        echo -n "Selecciona el número del entorno virtual a activar: "
        read opt
        if [[ "$opt" =~ ^[0-9]+$ ]] && [ "$opt" -ge 1 ] && [ "$opt" -le "${#virtualenvs[@]}" ]; then
            env="${virtualenvs[$((opt-1))]}"
            activate_and_check "$env"
            break
        else
            echo "Opción inválida. Intenta de nuevo."
        fi
    done
fi
