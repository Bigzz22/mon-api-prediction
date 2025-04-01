import subprocess
import sys

def run_tests():
    print("Lancement des tests unitaires...")
    result = subprocess.run(['pytest', 'tests'], capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print("Des tests ont échoué. Arrêt du lancement de l'API.")
        sys.exit(1)
    print("Tous les tests sont passés. Démarrage de l'API.")

if __name__ == "__main__":
    run_tests()

    # Démarrage de l'API
    from app import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)