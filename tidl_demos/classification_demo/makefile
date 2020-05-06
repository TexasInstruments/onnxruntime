################################################################################
# Paths
ORTROOT=../..

################################################################################
# Build directories
OBJDIR = ./obj
OUTDIR = ./bin

################################################################################
# Compiler Config
CXX = g++
CXXFLAGS = -std=c++14 -g
INC  = -I$(ORTROOT)/include -I$(ORTROOT)/include/onnxruntime/core/session/ -I$(ORTROOT)/../../opencv/opencv-3.1.0/modules/core/include
INC += -I$(ORTROOT)/../../opencv/opencv-3.1.0/modules/highgui/include
INC += -I$(ORTROOT)/../../opencv/opencv-3.1.0/modules/imgcodecs/include
INC += -I$(ORTROOT)/../../opencv/opencv-3.1.0/modules/videoio/include
INC += -I$(ORTROOT)/../../opencv/opencv-3.1.0/modules/imgproc/include
CXXFLAGS += $(INC) -pthread -g

################################################################################
# Linker Config
LIBDIR = -L$(ORTROOT)/build/Linux/Debug/ -L$(ORTROOT)/../../opencv/opencv-3.1.0/cmake/lib -L$(ORTROOT)/../../opencv/opencv-3.1.0/cmake/3rdparty/lib
LIBS = -lonnxruntime

LIBS += -lopencv_imgcodecs
LIBS += -lopencv_imgproc
LIBS += -lopencv_core
LIBS += -llibtiff 
LIBS += -llibwebp
LIBS += -llibpng
LIBS += -llibjpeg
LIBS += -lIlmImf
LIBS += -lzlib
LIBS += -llibjasper


LDFLAGS = $(LIBDIR) $(LIBS) -ldl -pthread

################################################################################
# Srcs Etc
SRCS := main.cc validator.cc
OBJS := $(SRCS:%=$(OBJDIR)/%.o)

################################################################################
# Targets
all: $(OBJDIR) $(OUTDIR) $(OUTDIR)/classification_demo

$(OUTDIR)/classification_demo: $(OBJS) 
	$(CXX)  $(CXXFLAGS) $(OBJS)  $(LDFLAGS)  -o $@ 

# c++ source
$(OBJDIR)/%.cc.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ 

$(OBJDIR) $(OUTDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(OUTDIR)
