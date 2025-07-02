#!/usr/bin/env python3
"""
Setup script to help configure Azure ML config.json securely.
"""

import json
import os
import shutil
from pathlib import Path


def setup_config_json():
    """Guide user through setting up config.json securely."""
    print("Azure ML Configuration Setup")
    print("=" * 40)

    # Check if config.json already exists
    project_config = Path("./config.json")
    user_config = Path.home() / ".azureml_example" / "config.json"

    if project_config.exists():
        print(f"✅ Found config.json in project root: {project_config.absolute()}")
        return True

    if user_config.exists():
        print(f"✅ Found config.json in user directory: {user_config}")
        choice = input("Copy to project root? (y/n): ").lower().strip()
        if choice == "y":
            shutil.copy2(user_config, project_config)
            print(f"✅ Copied config.json to project root")
            set_secure_permissions(project_config)
            return True
        return True

    # No config found, guide user
    print("❌ No config.json found")
    print("\nTo get your config.json:")
    print("1. Go to Azure ML Studio: https://ml.azure.com")
    print("2. Select your workspace")
    print("3. Click the download icon (⬇️) next to your workspace name")
    print("4. Save the downloaded config.json file")

    file_path = input("\nEnter the path to your downloaded config.json (or press Enter to skip): ").strip()

    if file_path and os.path.exists(file_path):
        # Validate the config file
        if validate_config_json(file_path):
            shutil.copy2(file_path, project_config)
            print(f"✅ Copied config.json to project root")
            set_secure_permissions(project_config)
            return True
        else:
            print("❌ Invalid config.json file")
            return False

    print("⚠️  Setup incomplete. Please download config.json from Azure ML Studio")
    return False


def validate_config_json(file_path):
    """Validate that the config.json file has required fields."""
    try:
        with open(file_path) as f:
            config = json.load(f)

        required_fields = ["subscription_id", "resource_group", "workspace_name"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False

        print("✅ Valid config.json file")
        print(f"   Workspace: {config['workspace_name']}")
        print(f"   Resource Group: {config['resource_group']}")
        print(f"   Subscription: {config['subscription_id']}")
        return True

    except json.JSONDecodeError:
        print("❌ Invalid JSON format")
        return False
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False


def set_secure_permissions(file_path):
    """Set secure file permissions (Unix/macOS only)."""
    try:
        # Set read/write for owner only (600)
        os.chmod(file_path, 0o600)
        print("✅ Set secure file permissions (600)")
    except Exception as e:
        print(f"⚠️  Could not set file permissions: {e}")


def create_gitignore_entry():
    """Ensure config.json is in .gitignore."""
    gitignore_path = Path(".gitignore")

    if not gitignore_path.exists():
        print("⚠️  No .gitignore found, creating one")
        with open(gitignore_path, "w") as f:
            f.write("config.json\n")
        return

    with open(gitignore_path) as f:
        content = f.read()

    if "config.json" not in content:
        print("⚠️  Adding config.json to .gitignore")
        with open(gitignore_path, "a") as f:
            f.write("\nconfig.json\n")
    else:
        print("✅ config.json already in .gitignore")


def main():
    """Run setup."""
    success = setup_config_json()
    create_gitignore_entry()

    if success:
        print("\n" + "=" * 40)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Run the test: python azureml_example/tests/test_azureml.py")
        print("2. Start developing with Azure ML")
    else:
        print("\n" + "=" * 40)
        print("❌ Setup incomplete")
        print("Please download config.json from Azure ML Studio")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
