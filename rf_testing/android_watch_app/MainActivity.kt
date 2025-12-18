package com.example.puckremote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material.*
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            // MaterialTheme provides the standard Wear OS look
            MaterialTheme {
                PuckApp()
            }
        }
    }
}

@Composable
fun PuckApp() {
    // "CoroutineScope" allows us to run network tasks off the main thread
    val scope = rememberCoroutineScope()
    
    // "State" to show user feedback (e.g. "Sent!", "Error")
    var statusText by remember { mutableStateOf("Ready") }

    // Column: Stacks items vertically
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black), // Watch apps are usually black bg
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        
        Text(text = "Outlet 1", color = Color.Gray, style = MaterialTheme.typography.caption1)
        
        Spacer(modifier = Modifier.height(8.dp))

        // Row: Places the ON and OFF buttons side by side
        Row(
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // ON BUTTON
            BigButton(text = "ON", color = Color(0xFF4CAF50)) {
                statusText = "Sending ON..."
                scope.launch {
                    sendSignal("1 ON", onSuccess = { statusText = "ON Sent!" }, onError = { statusText = "Error!" })
                }
            }
            
            // OFF BUTTON
            BigButton(text = "OFF", color = Color(0xFFF44336)) {
                statusText = "Sending OFF..."
                scope.launch {
                    sendSignal("1 OFF", onSuccess = { statusText = "OFF Sent!" }, onError = { statusText = "Error!" })
                }
            }
        }
        
        Spacer(modifier = Modifier.height(12.dp))
        
        Text(text = statusText, color = Color.White)
    }
}

// Helper function to keep our UI code clean
@Composable
fun BigButton(text: String, color: Color, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(backgroundColor = color),
        modifier = Modifier.size(60.dp) // Big circular touch target
    ) {
        Text(text = text, style = MaterialTheme.typography.button)
    }
}

// The actual network logic
suspend fun sendSignal(btnName: String, onSuccess: () -> Unit, onError: () -> Unit) {
    try {
        val response = RetrofitClient.api.triggerButton(ControlRequest(btnName))
        // If code reaches here, it succeeded (200 OK)
        onSuccess()
    } catch (e: Exception) {
        e.printStackTrace()
        onError()
    }
}
