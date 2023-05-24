NOTEBOOKS_DIR := notebooks
OUTPUT_FORMAT := script

.PHONY: run

install_requirements:
	@pip install -r requirements.txt

run:
	@jupyter nbconvert --execute $(NOTEBOOKS_DIR)/$(NOTEBOOK) --to $(OUTPUT_FORMAT) --output /dev/null
