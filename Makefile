CC = @gcc
CFLAGS = -msse3 -msse4.1 -Wall -std=c99

PROGNAME = simdsort
OBJ = main.o sort_simd.o
OBJDIR = obj
VPATH = src

$(PROGNAME): $(addprefix $(OBJDIR)/, $(OBJ))
	@echo -n Creando ejecutable...
	$(CC) $(CFLAGS) -o $@ $^
	@echo ok

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	@echo -n Compilando «$<» ...
	$(CC) $(CFLAGS) -c $< -o $@
	@echo ok

$(OBJDIR):
	@echo -n Creando directorio del codigo objeto... 
	@mkdir -p $(OBJDIR)
	@echo ok
# dependencias

$(OBJDIR)/main.o: main.h
$(OBJDIR)/sort_simd.o: sort_simd.h

clean:
	@rm -f $(PROGNAME)
	@rm -rf $(OBJDIR)
	@echo Todo limpio
