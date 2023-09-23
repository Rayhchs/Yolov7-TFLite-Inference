CC = g++
CFLAGS = -c -Wall
INCLUDE += -I/usr/local/include/tensorflow/
INCLUDE += -I/usr/local/include/flatbuffers/
INCLUDE += -I/usr/local/include/yaml-cpp/
LDFLAGS += -L/usr/local/lib -ltensorflow-lite -ldl -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lpthread
LDFLAGS += -lyaml-cpp

SOURCES = main.cpp model/model.cpp

OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS) 

.cpp.o:
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *o $(EXECUTABLE)


