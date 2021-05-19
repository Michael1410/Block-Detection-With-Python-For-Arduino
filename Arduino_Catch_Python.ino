
char Input[50];
int input_counter = 0;

float FBigx_coordinate = 0;
float FBigy_coordinate = 0;
float SBigx_coordinate = 0;
float SBigy_coordinate = 0;

float FSmallx_coordinate = 0;
float FSmally_coordinate = 0;
float SSmallx_coordinate = 0;
float SSmally_coordinate = 0;
float Robot_xcoordinate = 0;
float Robot_ycoordinate = 0;
float BotAngle = 0;
int Stopper = 0;

void setup() {
  // put your setup code here, to run once:
Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
    ReadSerial();
  
  

}


void ReadSerial()
{
  while(Serial.available() > 0)
  {
    Input[input_counter]=Serial.read();
    if(Input[input_counter]== '\n') //newline indicates end of a message
    {
      if(Input[input_counter-1]== 'x') //message type is x, so handle that message
      {
        FBigx_coordinate = atof(Input);
        Serial.print("First Big Lego x: ");
        Serial.println(FBigx_coordinate);
        delay(100);
      }
      if(Input[input_counter-1] == 'y')
      {
        FBigy_coordinate = atof(Input);
        Serial.print("First Big Lego y: ");
        Serial.println(FBigy_coordinate);
        delay(100);
      }
      if(Input[input_counter-1]== 'q') //message type is x, so handle that message
      {
        SBigx_coordinate = atof(Input);
        Serial.print("Second Big Lego x: ");
        Serial.println(SBigx_coordinate);
        delay(100);
      }
      if(Input[input_counter-1] == 'w')
      {
        SBigy_coordinate = atof(Input);
        Serial.print("Second Big Lego y: ");
        Serial.println(SBigy_coordinate);
        delay(100);
      }
      if(Input[input_counter-1]== 'e') //message type is x, so handle that message
      {
        FSmallx_coordinate = atof(Input);
        Serial.print("First Small Lego x: ");
        Serial.println(FSmallx_coordinate);
        delay(100);
      }
      if(Input[input_counter-1] == 'r')
      {
        FSmally_coordinate = atof(Input);
        Serial.print("First Small Lego y: ");
        Serial.println(FSmally_coordinate);
        delay(100);
      }

       if(Input[input_counter-1]== 't') //message type is x, so handle that message
      {
        SSmallx_coordinate = atof(Input);
        Serial.print("Second Small Lego x: ");
        Serial.println(SSmallx_coordinate);
        delay(100);
      }
      if(Input[input_counter-1] == 'y')
      {
        SSmally_coordinate = atof(Input);
        Serial.print("Second Small Lego y: ");
        Serial.println(SSmally_coordinate);
        delay(100);
        
      }
      if(Input[input_counter-1] == 'u')
      {
        Robot_xcoordinate = atof(Input);
        Serial.print("Robot x: ");
        Serial.println(Robot_xcoordinate);
        delay(100);
        
      }
  if(Input[input_counter-1] == 'i')
      {
        Robot_ycoordinate = atof(Input);
        Serial.print("Robot x: ");
        Serial.println(Robot_ycoordinate);
        delay(100);
        
      }
        if(Input[input_counter-1] == 'o')
      {
        float BotAngle = atof(Input);
        Serial.print("Robot x: ");
        Serial.println(BotAngle);
        delay(100);
        
      }
      input_counter = 0;
    }
    else
    {
      input_counter++;
      
    }
    
  }
}
