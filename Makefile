CC = gcc
CFLAGS = -msse3 -msse4.1 -Wall -std=c99

PROGNAME = simdsort
OBJ = main.o sort_simd.o
OBJDIR = build/obj
VPATH = src

$(PROGNAME): $(addprefix $(OBJDIR)/, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)
# dependencias

$(OBJDIR)/main.o: main.h
$(OBJDIR)/sort_simd.o: sort_simd.h

clean:
	rm -f $(PROGNAME)
	rm -rf $(OBJDIR)
